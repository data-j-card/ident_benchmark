import numpy as np
from scipy import stats

REPS = 20_000
SEED = 1
GOOD = 0.03


def main():
    rng = np.random.default_rng(SEED)
    for p, caida in [(0.10, -0.08), (0.05, -0.30), (0.03, -0.40)]:
        truth = (1 - p) * GOOD + p * caida
        for g in (10, 20, 30, 50):
            collapsed = rng.uniform(size=(REPS, g)) < p
            d = np.where(collapsed, caida, GOOD) + rng.normal(0, 0.02, (REPS, g))
            mean = d.mean(axis=1)
            half = stats.t.ppf(0.975, g - 1) * d.std(axis=1, ddof=1) / np.sqrt(g)
            cov = np.mean((mean - half <= truth) & (truth <= mean + half))
            none = np.mean(~collapsed.any(axis=1))
            print(
                f"colapso {p:.0%} con caida {caida:+.2f}  G={g:<3} "
                f"cobertura t(G-1)={cov:.3f}  muestras sin ningun colapso={none:.0%}"
            )


if __name__ == "__main__":
    main()
