from collections import Counter
from dataclasses import dataclass

import numpy as np

from benchid.identity import anchor, l2_normalize
from benchid.splits import view


@dataclass(frozen=True)
class Scores:
    genuine: np.ndarray
    gen_id: np.ndarray
    impostor: np.ndarray
    imp_a: np.ndarray
    imp_b: np.ndarray


def pair_scores(emb: np.ndarray, identity: np.ndarray) -> Scores:
    emb = l2_normalize(emb)
    ids = np.asarray(identity)
    i, j = np.triu_indices(len(ids), k=1)
    s = np.einsum("kd,kd->k", emb[i], emb[j])
    same = ids[i] == ids[j]
    return Scores(s[same], ids[i][same], s[~same], ids[i][~same], ids[j][~same])


def anchor_inputs(d: dict[str, np.ndarray]):
    views = [view(p) for p in d["photo"]]
    in_gallery_column = np.array([v.in_gallery_column for v in views])
    is_gallery = np.array([v.is_gallery for v in views])
    gallery = {int(i): d["embedding"][is_gallery & (d["identity"] == i)] for i in np.unique(d["identity"])}
    gallery = {i: g for i, g in gallery.items() if len(g)}
    return d["embedding"], d["identity"], ~in_gallery_column, gallery


def anchor_scores(
    probe_emb: np.ndarray,
    probe_id: np.ndarray,
    probe_genuine: np.ndarray,
    gallery: dict[int, np.ndarray],
) -> Scores:
    probe_emb = l2_normalize(probe_emb)
    probe_id = np.asarray(probe_id)
    labels = sorted(gallery)
    anchors = np.stack([anchor(gallery[g]) for g in labels])
    s = probe_emb @ anchors.T
    anchor_id = np.broadcast_to(np.asarray(labels), s.shape)
    probe = np.broadcast_to(probe_id[:, None], s.shape)
    own = anchor_id == probe
    gen = own & np.asarray(probe_genuine)[:, None]
    return Scores(s[gen], probe[gen], s[~own], anchor_id[~own], probe[~own])


def tau_at_far(impostor: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    s = np.asarray(impostor, dtype=np.float64)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    return _tau_sorted(s[order], w[order], alpha)


def _tau_sorted(s_desc: np.ndarray, w_desc: np.ndarray, alpha: float) -> float:
    cum = np.cumsum(w_desc)
    fitting = int(np.searchsorted(cum, alpha * cum[-1], side="right"))
    if fitting >= len(s_desc):
        return -np.inf
    return float(np.nextafter(s_desc[fitting], np.inf))


def far(impostor: np.ndarray, tau: float) -> float:
    return float(np.mean(np.asarray(impostor) >= tau))


def tpr(genuine: np.ndarray, tau: float) -> float:
    return float(np.mean(np.asarray(genuine) >= tau))


@dataclass(frozen=True)
class OperatingPoint:
    alpha: float
    tau: float
    tau_low: float
    tau_high: float
    far: float
    tpr: float
    tpr_low: float
    tpr_high: float
    n_impostor: int
    n_genuine: int
    n_identities: int
    boot_taus: np.ndarray


def operating_point(
    scores: Scores,
    alpha: float,
    n_boot: int = 2000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> OperatingPoint:
    rng = rng if rng is not None else np.random.default_rng()
    labels = np.unique(np.concatenate([scores.gen_id, scores.imp_a, scores.imp_b]))
    pos = {g: k for k, g in enumerate(labels)}
    ia = np.array([pos[g] for g in scores.imp_a])
    ib = np.array([pos[g] for g in scores.imp_b])
    ig = np.array([pos[g] for g in scores.gen_id])

    order = np.argsort(-scores.impostor, kind="stable")
    s_desc, ia, ib = scores.impostor[order], ia[order], ib[order]

    tau = _tau_sorted(s_desc, np.ones_like(s_desc), alpha)
    taus, tprs = np.empty(n_boot), np.empty(n_boot)
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels)).astype(float)
        taus[r] = _tau_sorted(s_desc, w[ia] * w[ib], alpha)
        wg = w[ig]
        tprs[r] = np.sum(wg * (scores.genuine >= taus[r])) / np.sum(wg)
    q = [(1 - level) / 2, (1 + level) / 2]
    t_lo, t_hi = np.quantile(taus, q)
    p_lo, p_hi = np.quantile(tprs, q)
    return OperatingPoint(
        alpha, tau, float(t_lo), float(t_hi),
        far(scores.impostor, tau), tpr(scores.genuine, tau), float(p_lo), float(p_hi),
        len(scores.impostor), len(scores.genuine), len(labels), taus,
    )


def false_accepts_by_pair(scores: Scores, tau: float) -> list[tuple[tuple, int]]:
    hit = scores.impostor >= tau
    return Counter(tuple(sorted(p)) for p in zip(scores.imp_a[hit], scores.imp_b[hit])).most_common()
