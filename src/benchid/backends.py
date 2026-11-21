import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from benchid.jobs import GenJob, TrainSpec


@dataclass(frozen=True)
class TrainItem:
    target: Path
    text: str
    controls: tuple[Path, ...]


@dataclass(frozen=True)
class TrainResult:
    checkpoints: dict[int, Path]
    loss: tuple[float, ...]
    seconds: float
    vram_gb: float


@dataclass(frozen=True)
class GenResult:
    seconds: float
    vram_gb: float


class Backend(Protocol):
    name: str

    def train(self, spec: TrainSpec, items: tuple[TrainItem, ...], out: Path) -> TrainResult: ...

    def generate(
        self, job: GenJob, prompt: str, refs: tuple[Path, ...], lora: Path | None, size: tuple[int, int], out: Path
    ) -> GenResult: ...


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


class FakeBackend:
    name = "falso"

    def train(self, spec: TrainSpec, items: tuple[TrainItem, ...], out: Path) -> TrainResult:
        rng = _rng(spec.id)
        loss = tuple(float(0.3 * np.exp(-e / 6) + 0.05 + 0.01 * rng.random()) for e in range(1, spec.epochs + 1))
        checkpoints = {}
        for epoch in range(1, spec.epochs + 1):
            path = out / f"epoch-{epoch:03d}.safetensors"
            path.write_bytes(rng.bytes(64))
            checkpoints[epoch] = path
        return TrainResult(checkpoints, loss, seconds=60.0 * len(items), vram_gb=42.0)

    def generate(
        self, job: GenJob, prompt: str, refs: tuple[Path, ...], lora: Path | None, size: tuple[int, int], out: Path
    ) -> GenResult:
        pixels = _rng(replace(job, repeat=0).id).integers(0, 256, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
        Image.fromarray(pixels).resize(size, Image.NEAREST).save(out, format="PNG")
        return GenResult(seconds=30.0, vram_gb=58.0)
