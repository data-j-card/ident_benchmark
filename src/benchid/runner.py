import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from benchid.arms import ARMS, arm_references, qwen_training_pairs
from benchid.backends import Backend, TrainItem
from benchid.captions import training_text
from benchid.cost import usd_from_gpu_seconds
from benchid.jobs import GenJob, TrainSpec
from benchid.plan import Phase, Pilot
from benchid.splits import Split


class BudgetExhausted(RuntimeError):
    pass


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class Runner:
    def __init__(
        self,
        pilot: Pilot,
        backend: Backend,
        out: Path,
        split_for: Callable[[str], Split],
        photos: Path,
        tracker=None,
    ) -> None:
        self.pilot, self.backend, self.out, self.split_for, self.photos = pilot, backend, out, split_for, photos
        self.tracker = tracker
        self.train_dir, self.image_dir = out / "entrenamientos", out / "imagenes"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def spent_usd(self) -> float:
        records = [*self.train_dir.glob("*/resultado.json"), *self.image_dir.glob("*.json")]
        seconds = sum(json.loads(p.read_text(encoding="utf-8"))["seconds"] for p in records)
        return usd_from_gpu_seconds(seconds, self.pilot.raw["usd_por_hora"])

    def _check_budget(self) -> None:
        spent, cap = self.spent_usd(), self.pilot.raw["tope_usd"]
        if spent >= cap:
            raise BudgetExhausted(f"spent USD {spent:.2f} of a USD {cap} cap")

    def run(self, phase: Phase) -> None:
        for spec in phase.trains:
            self._train(phase.name, spec)
        for job in phase.gens:
            self._generate(phase.name, job)

    def _train_items(self, spec: TrainSpec) -> tuple[TrainItem, ...]:
        split = self.split_for(spec.identity)
        subject = self.pilot.train_subjects[spec.model]
        if spec.model == "qwen":
            return tuple(
                TrainItem(self.photos / p.target, training_text(subject, p.target, spec.caption),
                          tuple(self.photos / c for c in p.controls))
                for p in qwen_training_pairs(split)
            )
        return tuple(TrainItem(self.photos / p, training_text(subject, p, spec.caption), ()) for p in split.inputs)

    def _train(self, phase: str, spec: TrainSpec) -> None:
        folder = self.train_dir / spec.id
        if (folder / "resultado.json").exists():
            return
        self._check_budget()
        folder.mkdir(exist_ok=True)
        result = self.backend.train(spec, self._train_items(spec), folder)
        record = {"phase": phase, "spec": asdict(spec), "backend": self.backend.name, "loss": list(result.loss),
                  "seconds": result.seconds, "vram_gb": result.vram_gb,
                  "checkpoints": {e: p.name for e, p in result.checkpoints.items()}}
        _write_json(folder / "resultado.json", record)
        if self.tracker:
            self.tracker.train(spec, record)

    def _generate(self, phase: str, job: GenJob) -> None:
        image, meta = self.image_dir / f"{job.id}.png", self.image_dir / f"{job.id}.json"
        if meta.exists():
            return
        self._check_budget()
        lora = None
        if job.train_id:
            trained = _read(self.train_dir / job.train_id / "resultado.json")
            lora = self.train_dir / job.train_id / trained["checkpoints"][str(job.epoch)]
        refs = tuple(self.photos / p for p in arm_references(ARMS[job.arm], self.split_for(job.identity)))
        prompt = self.pilot.prompt(job)
        tmp = image.with_suffix(".tmp.png")
        result = self.backend.generate(job, prompt, refs, lora, self.pilot.size, tmp)
        os.replace(tmp, image)
        record = {"phase": phase, "job": asdict(job), "prompt": prompt, "backend": self.backend.name,
                  "refs": [r.name for r in refs], "seconds": result.seconds, "vram_gb": result.vram_gb}
        _write_json(meta, record)
        if self.tracker:
            self.tracker.image(job, record, image)
