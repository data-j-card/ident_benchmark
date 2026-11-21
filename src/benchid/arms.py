from dataclasses import dataclass
from typing import Literal

from benchid.splits import Split, references

Model = Literal["qwen", "hidream"]
N_REFS = 3


@dataclass(frozen=True)
class Arm:
    name: str
    model: Model
    lora: bool
    n_refs: int


ARMS = {
    arm.name: arm
    for arm in (
        Arm("qwen_referencias", "qwen", lora=False, n_refs=N_REFS),
        Arm("qwen_lora", "qwen", lora=True, n_refs=N_REFS),
        Arm("hidream_referencias", "hidream", lora=False, n_refs=1),
        Arm("hidream_lora", "hidream", lora=True, n_refs=0),
    )
}


@dataclass(frozen=True)
class TrainPair:
    target: str
    controls: tuple[str, ...]


def arm_references(arm: Arm, split: Split) -> tuple[str, ...]:
    return references(split, arm.n_refs) if arm.n_refs else ()


def qwen_training_pairs(split: Split) -> tuple[TrainPair, ...]:
    refs = references(split, N_REFS)
    spare = next(p for p in split.inputs if p not in refs)
    return tuple(
        TrainPair(target, tuple(p for p in refs if p != target) + (spare,) if target in refs else refs)
        for target in split.inputs
    )
