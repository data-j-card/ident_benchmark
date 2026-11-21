from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    point: float
    low: float
    high: float
    n_identities: int


def per_identity_means(values: np.ndarray, identity: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(identity, return_inverse=True)
    return np.bincount(inverse, weights=values) / np.bincount(inverse)


def cluster_bootstrap_mean(
    values: np.ndarray,
    identity: np.ndarray,
    n_boot: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Estimate:
    rng = rng if rng is not None else np.random.default_rng()
    means = per_identity_means(np.asarray(values, dtype=np.float64), np.asarray(identity))
    return _bootstrap_of_means(means, n_boot, level, rng)


def paired_difference(
    a: np.ndarray,
    b: np.ndarray,
    identity_a: np.ndarray,
    identity_b: np.ndarray,
    n_boot: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Estimate:
    rng = rng if rng is not None else np.random.default_rng()
    d = identity_differences(a, b, identity_a, identity_b)
    return _bootstrap_of_means(d, n_boot, level, rng)


def identity_differences(
    a: np.ndarray, b: np.ndarray, identity_a: np.ndarray, identity_b: np.ndarray
) -> np.ndarray:
    if set(np.unique(identity_a)) != set(np.unique(identity_b)):
        raise ValueError("both arms must cover the same identities")
    return per_identity_means(np.asarray(a, float), np.asarray(identity_a)) - per_identity_means(
        np.asarray(b, float), np.asarray(identity_b)
    )


def t_interval_of_means(means: np.ndarray, level: float = 0.95) -> Estimate:
    means = np.asarray(means, dtype=np.float64)
    g = means.size
    if g < 2:
        raise ValueError("need at least two identities")
    half = stats.t.ppf(0.5 + level / 2, g - 1) * means.std(ddof=1) / np.sqrt(g)
    return Estimate(float(means.mean()), float(means.mean() - half), float(means.mean() + half), g)


def paired_difference_t(
    a: np.ndarray,
    b: np.ndarray,
    identity_a: np.ndarray,
    identity_b: np.ndarray,
    level: float = 0.95,
) -> Estimate:
    return t_interval_of_means(identity_differences(a, b, identity_a, identity_b), level)


def _bootstrap_of_means(
    means: np.ndarray, n_boot: int, level: float, rng: np.random.Generator
) -> Estimate:
    k = means.size
    if k < 2:
        raise ValueError("need at least two identities")
    boot = means[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
    alpha = 1 - level
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(means.mean()), float(low), float(high), k)


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    ci = stats.binomtest(successes, n).proportion_ci(confidence_level=level, method="wilson")
    return float(ci.low), float(ci.high)


def icc1(values: np.ndarray, identity: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels, inverse = np.unique(identity, return_inverse=True)
    k, big_n = labels.size, values.size
    n_i = np.bincount(inverse)
    means = np.bincount(inverse, weights=values) / n_i
    grand = values.mean()
    msb = np.sum(n_i * (means - grand) ** 2) / (k - 1)
    msw = np.sum((values - means[inverse]) ** 2) / (big_n - k)
    n0 = (big_n - np.sum(n_i**2) / big_n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))
