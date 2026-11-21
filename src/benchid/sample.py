import csv

import numpy as np

from benchid.paths import BANCO, EMBEDDINGS


def _bank() -> list[dict[str, str]]:
    with BANCO.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bank_ids() -> list[int]:
    return sorted(int(r["id"]) for r in _bank())


def excluded_ids() -> list[int]:
    return sorted(int(r["id"]) for r in _bank() if r["excluir"].strip().lower() == "si")


def embeddings() -> dict[str, np.ndarray]:
    parts = [np.load(p) for p in sorted(EMBEDDINGS.glob("*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def usable(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = ~np.isin(d["identity"], excluded_ids()) & d["detected"]
    return {k: v[keep] for k, v in d.items()}
