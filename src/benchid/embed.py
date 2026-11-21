import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL = "antelopev2"
DET_SIZE = (640, 640)


@dataclass(frozen=True)
class Face:
    detected: bool
    det_score: float
    box_w: int
    box_h: int
    embedding: np.ndarray


class Embedder:
    def __init__(self, providers: tuple[str, ...] = ("CPUExecutionProvider",)):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=MODEL, allowed_modules=["detection", "recognition"], providers=list(providers))
        self.app.prepare(ctx_id=-1, det_size=DET_SIZE)

    def __call__(self, bgr: np.ndarray) -> Face:
        faces = self.app.get(bgr)
        if not faces:
            return Face(False, 0.0, 0, 0, np.full(512, np.nan, dtype=np.float32))
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x0, y0, x1, y1 = f.bbox
        return Face(True, float(f.det_score), int(x1 - x0), int(y1 - y0), f.normed_embedding.astype(np.float32))


def versions(model_root: Path = Path.home() / ".insightface" / "models" / MODEL) -> dict:
    import insightface
    import onnxruntime

    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model_root.glob("*.onnx"))}
    return {
        "model": MODEL,
        "det_size": DET_SIZE,
        "insightface": insightface.__version__,
        "onnxruntime": onnxruntime.__version__,
        "sha256": hashes,
    }


def dump_versions(path: Path) -> None:
    path.write_text(json.dumps(versions(), indent=2), encoding="utf-8")
