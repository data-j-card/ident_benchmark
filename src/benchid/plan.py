import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from benchid.arms import ARMS, Model
from benchid.jobs import GenJob, TrainSpec
from benchid.paths import CONFIGS

MODELS: tuple[Model, ...] = ("qwen", "hidream")
COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Weight:
    repo: str
    revision: str
    file: str

    def __post_init__(self) -> None:
        if not COMMIT.fullmatch(self.revision):
            raise ValueError(f"{self.repo}: revision must be a full 40-hex commit, got {self.revision!r}")

    @property
    def ref(self) -> str:
        return f"{self.repo}@{self.revision}/{self.file}"


@dataclass(frozen=True)
class Phase:
    name: str
    trains: tuple[TrainSpec, ...]
    gens: tuple[GenJob, ...]


@dataclass(frozen=True)
class Pilot:
    raw: dict
    scenarios: dict[str, str]
    subjects: dict[str, str]
    train_subjects: dict[str, str]
    development: tuple[str, ...]
    weights: dict[Model, dict[str, Weight]]

    @classmethod
    def load(cls, configs: Path = CONFIGS) -> "Pilot":
        raw = tomllib.loads((configs / "piloto.toml").read_text(encoding="utf-8"))
        prompts = tomllib.loads((configs / "escenarios.toml").read_text(encoding="utf-8"))
        ids = json.loads((configs / "identidades.json").read_text(encoding="utf-8"))
        fill = lambda d: {k: v.format(token=raw["token"]) for k, v in d.items()}
        return cls(
            raw=raw,
            scenarios={s["id"]: s["texto"] for s in prompts["escenario"]},
            subjects=fill(prompts["sujeto"]),
            train_subjects=fill(prompts["entrenamiento"]),
            development=tuple(str(i) for i in ids["desarrollo"]),
            weights={m: {role: Weight(**w) for role, w in raw["pesos"][m].items()} for m in MODELS},
        )

    @property
    def size(self) -> tuple[int, int]:
        width, height = self.raw["lado"]
        return width, height

    def prompt(self, job: GenJob) -> str:
        return self.subjects[job.arm] + self.scenarios[job.scenario]

    def phase(self, name: str) -> Phase:
        cfg = self.raw["fases"][name]
        trains = self._trains(name, cfg)
        if name == "nativos":
            gens = tuple(
                GenJob(arm, self._weights(ARMS[arm].model), identity, scenario, self.raw["semilla"], repeat)
                for arm in cfg["brazos"]
                for identity in self.development
                for scenario in self.scenarios
                for repeat in range(self.raw["repeticiones"] + 1)
            )
        else:
            epochs = cfg.get("epocas_evaluadas", [self.raw["entrenamiento_base"]["epochs"]])
            gens = tuple(
                GenJob(f"{spec.model}_lora", spec.weights, spec.identity, scenario, self.raw["semilla"],
                       train_id=spec.id, epoch=epoch)
                for spec in trains
                for epoch in epochs
                for scenario in self.scenarios
            )
        return Phase(name, trains, gens)

    def _weights(self, model: Model) -> str:
        return ";".join(f"{role}={w.ref}" for role, w in sorted(self.weights[model].items()))

    def _trains(self, name: str, cfg: dict) -> tuple[TrainSpec, ...]:
        base = self.raw["entrenamiento_base"]
        ids = self.development[: cfg.get("identidades", 0)]
        spec = lambda model, identity, caption, rank, lr: TrainSpec(
            model, self._weights(model), identity, caption, rank, lr, base["epochs"]
        )
        if name == "nativos":
            return ()
        if name == "sonda_hidream":
            return tuple(spec("hidream", i, "A", base["rank"], base["lr"]) for i in ids)
        if name == "captions":
            return tuple(
                spec(m, i, c, base["rank"], base["lr"]) for m in MODELS for i in ids for c in ("A", "B")
            )
        if name == "hiper":
            if "caption" not in cfg:
                raise ValueError("fases.hiper.caption is unset: choose it from the captions phase first")
            return tuple(
                spec(m, i, cfg["caption"], r, lr)
                for m in MODELS for i in ids for r in cfg["rangos"] for lr in cfg["tasas"]
            )
        raise ValueError(f"unknown phase {name!r}")
