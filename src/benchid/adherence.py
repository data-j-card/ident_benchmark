import hashlib
import json
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from benchid.paths import CONFIGS, REPO
from benchid.plan import require_commit
from benchid.runner import write_json

IMAGE = "<image>"
IMAGE_TOKEN_INDEX = -200


@dataclass(frozen=True)
class Snapshot:
    repo: str
    revision: str
    dir: Path
    sha256: dict[str, str]

    def __post_init__(self) -> None:
        require_commit(self.repo, self.revision)


def verify(snapshot: Snapshot) -> None:
    for name, expected in sorted(snapshot.sha256.items()):
        path = snapshot.dir / name
        if not path.is_file():
            raise ValueError(f"{snapshot.repo}: missing {path}")
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{snapshot.repo}: {name} has sha256 {actual}, pinned {expected}")


@dataclass(frozen=True)
class JudgeConfig:
    question: str
    answer: str
    system: str
    subject: str
    model: Snapshot
    tokenizer: Snapshot
    vision: Snapshot
    device_map: dict[str, int | str]

    @classmethod
    def load(cls, configs: Path = CONFIGS, root: Path = REPO) -> "JudgeConfig":
        raw = tomllib.loads((configs / "juez.toml").read_text(encoding="utf-8"))
        snap = lambda s: Snapshot(s["repo"], s["revision"], root / s["dir"], dict(s["sha256"]))
        shots = raw["instantaneas"]
        return cls(
            question=raw["pregunta"],
            answer=raw["respuesta"],
            system=raw["sistema"],
            subject=raw["sujeto"],
            model=snap(shots["modelo"]),
            tokenizer=snap(shots["tokenizador"]),
            vision=snap(shots["vision"]),
            device_map=dict(raw["reparto"]),
        )

    @property
    def snapshots(self) -> tuple[Snapshot, ...]:
        return self.model, self.tokenizer, self.vision

    @property
    def revisions(self) -> dict[str, str]:
        return {s.repo: s.revision for s in self.snapshots}

    def judge_text(self, scenario_text: str) -> str:
        return self.subject + scenario_text

    def prompt(self, text: str) -> str:
        return f"{self.system} USER: {IMAGE}\n{self.question.format(text)} ASSISTANT: "


def score_run(out: Path, judge: Callable[[Path, str], float], cfg: JudgeConfig, scenarios: dict[str, str]) -> int:
    dest = out / "adherencia"
    dest.mkdir(parents=True, exist_ok=True)
    scored = 0
    for meta in sorted((out / "imagenes").glob("*.json")):
        target = dest / meta.name
        if target.exists():
            continue
        record = json.loads(meta.read_text(encoding="utf-8"))
        text = cfg.judge_text(scenarios[record["job"]["scenario"]])
        start = time.perf_counter()
        score = judge(meta.with_suffix(".png"), text)
        write_json(target, {"vqascore": score, "text": text, "seconds": time.perf_counter() - start,
                            "judge": cfg.revisions})
        scored += 1
    return scored


def _pad_square(image, background: tuple[int, ...]):
    from PIL import Image

    w, h = image.size
    side = max(w, h)
    canvas = Image.new(image.mode, (side, side), background)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


class Judge:
    def __init__(self, cfg: JudgeConfig):
        for snapshot in cfg.snapshots:
            verify(snapshot)
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from transformers import AutoTokenizer

        from clip_t5.model.language_model.clip_t5 import CLIPT5ForConditionalGeneration

        self.cfg, self.torch = cfg, torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.dir, use_fast=False, model_max_length=2048,
                                                       local_files_only=True)
        self.model = CLIPT5ForConditionalGeneration.from_pretrained(
            cfg.model.dir, mm_vision_tower=str(cfg.vision.dir), torch_dtype=torch.bfloat16,
            device_map=cfg.device_map, local_files_only=True,
        )
        tower = self.model.get_vision_tower()
        if not tower.is_loaded:
            tower.load_model()
        tower.to(device="cuda", dtype=torch.bfloat16)
        self.model.get_model().mm_projector.to(device="cuda", dtype=torch.bfloat16)
        self.model.eval().requires_grad_(False)
        self.processor = tower.image_processor
        self.background = tuple(int(x * 255) for x in self.processor.image_mean)

    def _ids(self, text: str):
        ids: list[int] = []
        for i, chunk in enumerate(text.split(IMAGE)):
            if i:
                ids.append(IMAGE_TOKEN_INDEX)
            ids.extend(self.tokenizer(chunk).input_ids)
        return self.torch.tensor([ids], device="cuda")

    def __call__(self, image_path: Path, text: str) -> float:
        from PIL import Image

        torch = self.torch
        image = _pad_square(Image.open(image_path).convert("RGB"), self.background)
        pixels = self.processor.preprocess(image, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
        input_ids, labels = self._ids(self.cfg.prompt(text)), self._ids(self.cfg.answer)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                             labels=labels, decoder_attention_mask=labels.ne(-100), images=pixels, return_dict=True)
        loss = torch.nn.functional.cross_entropy(out.logits[0].float(), labels[0].to(out.logits.device))
        return float(torch.exp(-loss))
