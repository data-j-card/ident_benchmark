import numpy as np
import pytest
from scipy import stats

from benchid.stats import (
    cluster_bootstrap_mean,
    icc1,
    paired_difference,
    paired_difference_t,
    per_identity_means,
    t_interval_of_means,
    wilson_interval,
)


def simulate(rng, k, m, mu=0.6, sd_between=0.1, sd_within=0.1):
    ident = np.repeat(np.arange(k), m)
    values = mu + rng.normal(0, sd_between, k)[ident] + rng.normal(0, sd_within, k * m)
    return values, ident


def test_each_identity_weighs_the_same():
    values = np.array([1.0, 1.0, 1.0, 0.0])
    ident = np.array([0, 0, 0, 1])
    np.testing.assert_allclose(per_identity_means(values, ident), [1.0, 0.0])
    assert cluster_bootstrap_mean(values, ident, rng=np.random.default_rng(0)).point == 0.5


def test_cluster_bootstrap_covers_the_true_mean_about_95_percent():
    rng = np.random.default_rng(42)
    reps, hits = 300, 0
    for _ in range(reps):
        values, ident = simulate(rng, k=30, m=20)
        est = cluster_bootstrap_mean(values, ident, n_boot=1000, rng=rng)
        hits += est.low <= 0.6 <= est.high
    assert 0.90 <= hits / reps <= 0.99


def test_naive_bootstrap_undercovers():
    rng = np.random.default_rng(42)
    reps, hits = 300, 0
    for _ in range(reps):
        values, _ = simulate(rng, k=30, m=20)
        boot = values[rng.integers(0, values.size, size=(1000, values.size))].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        hits += low <= 0.6 <= high
    assert hits / reps < 0.70


def test_paired_difference_recovers_a_known_effect():
    rng = np.random.default_rng(7)
    k, m = 50, 10
    ident = np.repeat(np.arange(k), m)
    person = rng.normal(0, 0.1, k)[ident]
    lora = 0.65 + person + rng.normal(0, 0.05, k * m)
    ref = 0.60 + person + rng.normal(0, 0.05, k * m)
    est = paired_difference(lora, ref, ident, ident, n_boot=2000, rng=rng)
    assert est.low <= 0.05 <= est.high
    assert est.high - est.low < 0.03


def test_paired_difference_rejects_unmatched_identities():
    with pytest.raises(ValueError):
        paired_difference(np.ones(2), np.ones(2), np.array([0, 1]), np.array([0, 2]))


def test_t_interval_matches_scipy_one_sample_t():
    d = np.random.default_rng(11).normal(0.03, 0.02, 30)
    ci = stats.ttest_1samp(d, 0).confidence_interval(0.95)
    est = t_interval_of_means(d)
    np.testing.assert_allclose((est.low, est.high), (ci.low, ci.high), atol=1e-12)


def test_paired_difference_t_recovers_a_known_effect():
    rng = np.random.default_rng(7)
    k, m = 50, 10
    ident = np.repeat(np.arange(k), m)
    person = rng.normal(0, 0.1, k)[ident]
    lora = 0.65 + person + rng.normal(0, 0.05, k * m)
    ref = 0.60 + person + rng.normal(0, 0.05, k * m)
    est = paired_difference_t(lora, ref, ident, ident)
    assert est.low <= 0.05 <= est.high
    assert est.n_identities == k


@pytest.mark.parametrize("x, n", [(0, 10), (7, 20), (95, 100)])
def test_wilson_matches_closed_form(x, n):
    z, p = 1.959963984540054, x / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    np.testing.assert_allclose(wilson_interval(x, n), (center - half, center + half), atol=1e-12)


@pytest.mark.parametrize("sd_between, expected", [(0.1, 0.5), (0.0, 0.0), (0.2, 0.8)])
def test_icc_recovers_simulated_value(sd_between, expected):
    values, ident = simulate(np.random.default_rng(3), k=400, m=15, sd_between=sd_between)
    assert icc1(values, ident) == pytest.approx(expected, abs=0.06)


def test_icc_handles_unequal_group_sizes():
    rng = np.random.default_rng(5)
    values, ident = simulate(rng, k=400, m=15)
    keep = rng.uniform(size=values.size) > 0.4
    assert icc1(values[keep], ident[keep]) == pytest.approx(0.5, abs=0.06)
