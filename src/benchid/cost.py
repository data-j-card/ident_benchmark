import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostInputs:
    train_cost: float
    gen_cost_lora: float
    gen_cost_ref: float


def cost_lora(c: CostInputs, n: float) -> float:
    return c.train_cost + n * c.gen_cost_lora


def cost_ref(c: CostInputs, n: float) -> float:
    return n * c.gen_cost_ref


def break_even_n(c: CostInputs) -> float:
    margin = c.gen_cost_ref - c.gen_cost_lora
    if margin <= 0:
        return math.inf
    return c.train_cost / margin


def usd_from_gpu_seconds(seconds: float, usd_per_hour: float) -> float:
    return seconds * usd_per_hour / 3600
