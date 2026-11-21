import hashlib
import json
from dataclasses import asdict, dataclass

from benchid.arms import ARMS, Model
from benchid.captions import Variant


def stable_id(obj: object) -> str:
    raw = json.dumps(asdict(obj), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class TrainSpec:
    model: Model
    weights: str
    identity: str
    caption: Variant
    rank: int
    lr: float
    epochs: int

    @property
    def id(self) -> str:
        return stable_id(self)


@dataclass(frozen=True)
class GenJob:
    arm: str
    weights: str
    identity: str
    scenario: str
    seed: int
    repeat: int = 0
    train_id: str | None = None
    epoch: int | None = None

    def __post_init__(self) -> None:
        if ARMS[self.arm].lora != (self.train_id is not None and self.epoch is not None):
            raise ValueError(f"{self.arm}: a LoRA arm needs train_id and epoch, a reference arm neither")

    @property
    def id(self) -> str:
        return stable_id(self)

    @property
    def config_key(self) -> str:
        return f"{self.arm}:{self.train_id}:{self.epoch}" if self.train_id else self.arm
