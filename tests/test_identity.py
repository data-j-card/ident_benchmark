import numpy as np
import pytest

from benchid.identity import anchor, identity_similarity, l2_normalize, match_rate, match_rate_curve


def test_same_face_has_similarity_one_and_orthogonal_zero():
    a = np.array([1.0, 0.0, 0.0])
    outputs = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(identity_similarity(outputs, a), [1.0, 0.0, -1.0])


def test_similarity_ignores_embedding_scale():
    rng = np.random.default_rng(0)
    v, a = rng.normal(size=(5, 512)), rng.normal(size=512)
    np.testing.assert_allclose(identity_similarity(v, a), identity_similarity(7 * v, 0.1 * a))


def test_anchor_is_unit_norm_and_not_dominated_by_a_large_embedding():
    enrollment = np.array([[1.0, 0.0], [0.0, 100.0]])
    a = anchor(enrollment)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    np.testing.assert_allclose(a, [np.sqrt(0.5), np.sqrt(0.5)])


def test_zero_embedding_is_an_error_not_a_silent_nan():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((1, 3)))


def test_match_rate_counts_ties_as_match():
    s = np.array([0.2, 0.5, 0.5, 0.9])
    assert match_rate(s, 0.5) == 0.75


def test_match_rate_curve_is_non_increasing_and_spans_one_to_zero():
    s = np.random.default_rng(1).uniform(-1, 1, size=200)
    curve = match_rate_curve(s, np.linspace(-1, 1.01, 50))
    assert curve[0] == 1.0 and curve[-1] == 0.0
    assert np.all(np.diff(curve) <= 0)
