import numpy as np
import pytest

from benchid.calibration import (
    anchor_scores,
    false_accepts_by_pair,
    far,
    operating_point,
    pair_scores,
    tau_at_far,
)


def test_tau_leaves_exactly_alpha_above():
    s = np.arange(1000) / 1000
    tau = tau_at_far(s, 0.01)
    assert far(s, tau) == pytest.approx(0.01)
    assert 0.989 < tau <= 0.990


def test_tau_is_the_smallest_that_meets_alpha():
    rng = np.random.default_rng(0)
    s = rng.normal(size=10_000)
    tau = tau_at_far(s, 1e-3)
    assert far(s, tau) <= 1e-3
    assert far(s, np.nextafter(tau, -np.inf)) > 1e-3


def test_ties_never_push_far_above_alpha():
    s = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    tau = tau_at_far(s, 0.2)
    assert far(s, tau) <= 0.2
    assert tau > 0.9


def test_weights_count_as_repeated_scores():
    s = np.array([0.9, 0.5, 0.1])
    w = np.array([1.0, 1.0, 8.0])
    repeated = np.array([0.9, 0.5] + [0.1] * 8)
    assert tau_at_far(s, 0.15, w) == tau_at_far(repeated, 0.15)


def test_pair_scores_splits_genuine_and_impostor():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sc = pair_scores(emb, np.array([1, 1, 2]))
    assert sc.genuine.tolist() == [1.0]
    assert sorted(sc.impostor.tolist()) == [0.0, 0.0]
    assert set(zip(sc.imp_a, sc.imp_b)) == {(1, 2)}


def test_anchor_scores_keeps_gallery_views_out_of_genuine():
    gallery = {1: np.array([[1.0, 0.0]]), 2: np.array([[0.0, 1.0]])}
    probes = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sc = anchor_scores(probes, np.array([1, 1, 2]), np.array([True, False, True]), gallery)
    assert sc.genuine.tolist() == [1.0, 1.0]
    assert len(sc.impostor) == 3
    assert np.all(sc.imp_a != sc.imp_b)


def _toy_scores(seed=0, n_id=30, per_id=6):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_id, 64))
    emb = np.repeat(centers, per_id, axis=0) + 0.6 * rng.normal(size=(n_id * per_id, 64))
    return pair_scores(emb, np.repeat(np.arange(n_id), per_id))


def test_operating_point_meets_alpha_and_brackets_tau():
    sc = _toy_scores()
    op = operating_point(sc, 1e-2, n_boot=200, rng=np.random.default_rng(1))
    assert op.far <= 1e-2
    assert op.tau_low <= op.tau <= op.tau_high
    assert op.tpr_low <= op.tpr <= op.tpr_high
    assert op.n_identities == 30


def test_false_accepts_are_grouped_by_identity_pair():
    sc = _toy_scores()
    tau = tau_at_far(sc.impostor, 1e-2)
    groups = false_accepts_by_pair(sc, tau)
    assert sum(n for _, n in groups) == int(np.sum(sc.impostor >= tau))
    assert all(a < b for (a, b), _ in groups)
