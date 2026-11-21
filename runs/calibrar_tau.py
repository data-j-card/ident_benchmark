import json
from pathlib import Path

import numpy as np

from benchid.calibration import Scores, anchor_inputs, anchor_scores, false_accepts_by_pair, operating_point
from benchid.identity import anchor, identity_similarity
from benchid.paths import CALIBRACION
from benchid.sample import embeddings, excluded_ids, usable
from benchid.splits import view
from benchid.stats import cluster_bootstrap_mean

ALPHAS = (1e-3, 1e-4)
N_BOOT = 10_000
SEED = 42


def detection_by_view(d: dict) -> dict[str, dict]:
    views = np.array([view(p).tag for p in d["photo"]])
    return {
        v: {"fotos": int((views == v).sum()), "detectadas": int(d["detected"][views == v].sum())}
        for v in sorted(set(views))
    }


def session_ceiling(d: dict, rng) -> dict:
    gallery = anchor_inputs(d)[3]
    is_input = np.array([view(p).is_input for p in d["photo"]])
    ids = d["identity"][is_input]
    s = np.array([identity_similarity(e, anchor(gallery[int(i)]))[0] for e, i in zip(d["embedding"][is_input], ids)])
    est = cluster_bootstrap_mean(s, ids, n_boot=N_BOOT, rng=rng)
    per = {int(i): float(s[ids == i].mean()) for i in np.unique(ids)}
    print(f"techo de sesion: {est.point:.3f} [{est.low:.3f}, {est.high:.3f}], {est.n_identities} identidades; "
          f"mas bajas {sorted(per.items(), key=lambda kv: kv[1])[:3]}")
    return {"media": est.point, "ic95": [est.low, est.high], "identidades": est.n_identities, "por_identidad": per}


def operating_points(scores: Scores, rng) -> tuple[list[dict], dict[str, np.ndarray]]:
    points, boot_taus = [], {}
    for a in ALPHAS:
        op = operating_point(scores, a, n_boot=N_BOOT, rng=rng)
        boot_taus[f"{a:.0e}"] = op.boot_taus
        fa = false_accepts_by_pair(scores, op.tau)
        points.append(
            {
                "alpha": a,
                "tau": op.tau, "tau_ic95": [op.tau_low, op.tau_high],
                "far_empirica": op.far, "falsas_aceptaciones": int(round(op.far * op.n_impostor)),
                "tpr": op.tpr, "tpr_ic95": [op.tpr_low, op.tpr_high],
                "parejas_con_falsa_aceptacion": len(fa),
                "top_parejas": [[list(map(int, p)), n] for p, n in fa[:10]],
            }
        )
        print(
            f"alpha={a:.0e}  tau={op.tau:.4f} [{op.tau_low:.4f}, {op.tau_high:.4f}]  "
            f"FAR={op.far:.2e} ({points[-1]['falsas_aceptaciones']} de {op.n_impostor})  "
            f"TPR={op.tpr:.4f} [{op.tpr_low:.4f}, {op.tpr_high:.4f}]  "
            f"parejas={len(fa)}  top={[(f'{a}-{b}', n) for (a, b), n in fa[:5]]}"
        )
    return points, boot_taus


def histogram(points: list[dict], scores: Scores, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bins = np.linspace(-0.3, 1.0, 131)
    ax.hist(scores.impostor, bins=bins, density=True, alpha=0.6, label="impostores")
    ax.hist(scores.genuine, bins=bins, density=True, alpha=0.6, label="genuinos")
    for p, ls in zip(points, ("--", ":")):
        ax.axvline(p["tau"], color="k", ls=ls, label=f"τ @ FAR {p['alpha']:.0e} = {p['tau']:.3f}")
    ax.set_xlabel("coseno ArcFace (antelopev2) con el ancla de galería")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)


def main() -> None:
    CALIBRACION.mkdir(parents=True, exist_ok=True)
    raw = embeddings()
    d = usable(raw)
    print(f"excluidas: {excluded_ids()}; fotos usadas: {len(d['photo'])}, identidades: {len(np.unique(d['identity']))}")

    rng = np.random.default_rng(SEED)
    scores = anchor_scores(*anchor_inputs(d))
    points, boot_taus = operating_points(scores, rng)
    results = {
        "genuinos": len(scores.genuine),
        "impostores": len(scores.impostor),
        "puntos": points,
        "techo_de_sesion": session_ceiling(d, rng),
        "deteccion_por_vista": detection_by_view(raw),
        "excluidas": excluded_ids(),
        "n_boot": N_BOOT,
        "semilla": SEED,
    }
    (CALIBRACION / "resultado.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    np.savez(CALIBRACION / "bootstrap_tau.npz", **boot_taus)
    histogram(points, scores, CALIBRACION / "histogramas.png")


if __name__ == "__main__":
    main()
