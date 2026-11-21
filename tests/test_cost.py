import math

import pytest

from benchid.cost import CostInputs, break_even_n, cost_lora, cost_ref, usd_from_gpu_seconds


def test_costs_cross_at_break_even():
    c = CostInputs(train_cost=2.0, gen_cost_lora=0.01, gen_cost_ref=0.03)
    n_star = break_even_n(c)
    assert n_star == pytest.approx(100.0)
    assert cost_lora(c, n_star) == pytest.approx(cost_ref(c, n_star))
    assert cost_lora(c, 50) > cost_ref(c, 50)
    assert cost_lora(c, 200) < cost_ref(c, 200)


def test_training_never_pays_off_if_references_are_not_dearer_per_image():
    assert break_even_n(CostInputs(2.0, 0.03, 0.03)) == math.inf
    assert break_even_n(CostInputs(2.0, 0.04, 0.03)) == math.inf


def test_gpu_seconds_to_usd():
    assert usd_from_gpu_seconds(3600, usd_per_hour=0.8) == pytest.approx(0.8)
