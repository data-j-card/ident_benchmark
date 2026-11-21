import json

import numpy as np

from benchid.paths import CONFIGS
from benchid.sample import bank_ids, excluded_ids

OUT = CONFIGS / "identidades.json"
SEED = 42
N_MEASURE = 30
N_DEV = 3


def main() -> None:
    eligible = sorted(set(bank_ids()) - set(excluded_ids()))
    order = np.random.default_rng(SEED).permutation(eligible).tolist()
    draw = {
        "semilla": SEED,
        "elegibles": len(eligible),
        "medicion": sorted(order[:N_MEASURE]),
        "desarrollo": sorted(order[N_MEASURE : N_MEASURE + N_DEV]),
        "reserva": sorted(order[N_MEASURE + N_DEV :]),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(draw, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(draw))


if __name__ == "__main__":
    main()
