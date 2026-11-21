import csv
from pathlib import Path

import numpy as np
from scipy import stats

E, K = 10, 3
DELTA = 0.03
REPS = 4000
N_BOOT = 1000
SEED = 20261005

SD_V = 0.03


def v_normal(rng, g):
    return rng.normal(0, SD_V, g)


def v_colapso(rng, g):
    p, bad = 0.10, -0.08
    good_mean = -p * bad / (1 - p)
    good_sd = np.sqrt((SD_V**2 - p * bad**2 - (1 - p) * good_mean**2) / (1 - p))
    collapse = rng.uniform(size=g) < p
    return np.where(collapse, bad, rng.normal(good_mean, good_sd, g))


def v_colas_pesadas(rng, g):
    return rng.standard_t(3, g) * SD_V / np.sqrt(3.0)


DISTRIBUCIONES = {"normal": v_normal, "colapso 10 %": v_colapso, "colas pesadas t3": v_colas_pesadas}


def simulate(rng, g, v_fn, alpha, gamma):
    u = rng.normal(0, 0.10, g)
    v = v_fn(rng, g)
    shape = (g, E, K)
    base = 0.55 + alpha[None, :, None] + u[:, None, None]
    ref = base + rng.normal(0, 0.05, (g, E, 1)) + rng.normal(0, 0.03, shape)
    lora = (
        base
        + (DELTA + v)[:, None, None]
        + gamma[None, :, None]
        + rng.normal(0, 0.05, (g, E, 1))
        + rng.normal(0, 0.03, shape)
    )
    return lora, ref


def intervals(rng, lora, ref):
    g = lora.shape[0]
    d = lora.mean(axis=(1, 2)) - ref.mean(axis=(1, 2))
    out = {}
    half = stats.t.ppf(0.975, g - 1) * d.std(ddof=1) / np.sqrt(g)
    out["t(G-1) por identidad"] = (d.mean() - half, d.mean() + half)
    boot = d[rng.integers(0, g, size=(N_BOOT, g))].mean(axis=1)
    out["bootstrap percentil por identidad"] = tuple(np.quantile(boot, [0.025, 0.975]))
    a, b = lora.ravel(), ref.ravel()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    z = stats.norm.ppf(0.975)
    out["ingenuo por imagen"] = (a.mean() - b.mean() - z * se, a.mean() - b.mean() + z * se)
    return out


def main():
    rng = np.random.default_rng(SEED)
    alpha = rng.normal(0, 0.05, E)
    gamma = rng.normal(0, 0.02, E)
    gamma -= gamma.mean()
    rows = []
    for dist_name, v_fn in DISTRIBUCIONES.items():
        for g in (10, 20, 30, 50):
            hits, widths = {}, {}
            for _ in range(REPS):
                lora, ref = simulate(rng, g, v_fn, alpha, gamma)
                for name, (lo, hi) in intervals(rng, lora, ref).items():
                    hits[name] = hits.get(name, 0) + (lo <= DELTA <= hi)
                    widths.setdefault(name, []).append(hi - lo)
            for name in hits:
                cov = hits[name] / REPS
                mc_se = np.sqrt(cov * (1 - cov) / REPS)
                rows.append(
                    {
                        "distribucion_v": dist_name,
                        "G": g,
                        "metodo": name,
                        "cobertura": round(cov, 4),
                        "error_montecarlo": round(mc_se, 4),
                        "ancho_medio": round(float(np.mean(widths[name])), 4),
                    }
                )
                print(f"{dist_name:18} G={g:<3} {name:34} cobertura={cov:.3f} ±{1.96 * mc_se:.3f}  ancho={np.mean(widths[name]):.4f}")
    out = Path(__file__).with_suffix(".csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n{REPS} réplicas por celda; semilla {SEED}; resultados en {out.name}")


if __name__ == "__main__":
    main()
