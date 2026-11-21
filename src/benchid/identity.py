import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding: the face was not embedded")
    return x / norms


def anchor(enrollment: np.ndarray) -> np.ndarray:
    enrollment = np.atleast_2d(enrollment)
    return l2_normalize(l2_normalize(enrollment).mean(axis=0))


def identity_similarity(outputs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return l2_normalize(np.atleast_2d(outputs)) @ l2_normalize(reference)


def match_rate(s_id: np.ndarray, tau: float) -> float:
    s_id = np.asarray(s_id, dtype=np.float64)
    if s_id.size == 0:
        raise ValueError("no outputs to aggregate")
    return float(np.mean(s_id >= tau))


def match_rate_curve(s_id: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return np.array([match_rate(s_id, t) for t in taus])
