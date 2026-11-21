import json

import numpy as np

from benchid.calibration import anchor_inputs, anchor_scores, false_accepts_by_pair
from benchid.paths import CALIBRACION
from benchid.sample import embeddings, usable
from benchid.splits import View

FIG = CALIBRACION / "figuras"
GEN, IMP = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GREY = "#c9c8c2"


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "legend.frameon": False,
    })
    return plt


def scores_and_ids():
    d = usable(embeddings())
    return anchor_scores(*anchor_inputs(d)), d


def f1_particion(plt):
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.set_axis_off()
    roles = {"entrada": GEN, "galeria": "#1baf7a", "sondeo": "#86b6ef", "fuera": GREY}

    def box(x, y, c, t):
        ax.add_patch(plt.Rectangle((x, y), 0.9, 0.9, color=c, lw=0))
        ax.text(x + 0.45, y + 0.45, t, ha="center", va="center", fontsize=7.5, color=INK)

    names = ["neutra", "sonrisa", "ojos\ncerrados", "sorpresa", "gafas"]
    for k in range(5):
        role = "entrada" if View(f"E{k + 1}", 0).is_input else "sondeo"
        box(k, 4.4, roles[role], f"E{k + 1}\n{names[k]}")
    for r in range(1, 5):
        for c in range(9):
            v = View(f"A{c}", r)
            role = "entrada" if v.is_input else "galeria" if v.is_gallery else "fuera" if v.in_gallery_column else "sondeo"
            box(c, 4 - r, roles[role], f"A{r},{c}")
    ax.text(-0.15, 4.85, "Primeros planos", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.text(-0.15, 1.95, "Multivista\n(fila r,\ncolumna c)", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.text(0, -0.35, "perfil", fontsize=8, color=INK2)
    ax.text(8.9, -0.35, "perfil", fontsize=8, color=INK2, ha="right")
    ax.text(4.45, -0.35, "frontal", fontsize=8, color=INK2, ha="center")
    labels = {
        "entrada": "Entrada del método (5 fotos)",
        "galeria": "Galería → ancla (7 fotos, fila 1)",
        "sondeo": "Sondeo genuino en la calibración",
        "fuera": "Solo impostora (casi duplicado de la galería)",
    }
    handles = [plt.Rectangle((0, 0), 1, 1, color=roles[k]) for k in labels]
    ax.legend(handles, labels.values(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8)
    ax.set_xlim(-1.6, 9.2)
    ax.set_ylim(-0.6, 5.4)
    ax.set_title("F1. Partición fija de las fotografías de una identidad")
    fig.savefig(FIG / "F1-particion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f2_histograma(plt, sc, tau):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-0.3, 1.0, 131)
    for s, c, lab in ((sc.impostor, IMP, "Personas distintas"), (sc.genuine, GEN, "Misma persona")):
        ax.hist(s, bins=bins, density=True, color=c, alpha=0.75, label=f"{lab} (n = {len(s):,})".replace(",", "."))
    ax.set_yscale("log")
    ax.axvline(tau, color=INK, lw=1.2, ls="--")
    ax.text(tau + 0.01, ax.get_ylim()[1] * 0.4, f"τ* = {tau:.3f}".replace(".", ","), fontsize=9, color=INK)
    ax.annotate("máx. impostor " + f"{sc.impostor.max():.3f}".replace(".", ","), (sc.impostor.max(), 0.05),
                xytext=(0.45, 0.6), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.annotate("mín. genuino " + f"{sc.genuine.min():.3f}".replace(".", ","), (sc.genuine.min(), 0.05),
                xytext=(0.5, 0.12), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel("Similitud coseno con el ancla (ArcFace, antelopev2)")
    ax.set_ylabel("Densidad (escala log)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("F2. Puntajes de la misma persona y de personas distintas")
    fig.savefig(FIG / "F2-histograma.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f3_far_tpr(plt, sc, tau, tau4):
    t = np.linspace(-0.1, 1.0, 1101)
    imp = np.sort(sc.impostor)
    gen = np.sort(sc.genuine)
    far = 1 - np.searchsorted(imp, t, side="left") / len(imp)
    tpr = 1 - np.searchsorted(gen, t, side="left") / len(gen)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    a1.plot(t, np.where(far > 0, far, np.nan), color=IMP, lw=2)
    for a, tt, ls in ((1e-3, tau, "--"), (1e-4, tau4, ":")):
        a1.axhline(a, color=INK2, lw=0.8, ls=ls)
        a1.axvline(tt, color=INK, lw=1, ls=ls)
        a1.text(tt + 0.01, a * 1.4, (f"α = 10⁻{'³⁴'[int(-np.log10(a)) - 3]} → τ = " + f"{tt:.3f}".replace(".", ",")), fontsize=8)
    a1.set_yscale("log")
    a1.set_xlim(-0.1, 0.5)
    a1.set_xlabel("Umbral τ")
    a1.set_ylabel("FAR(τ): impostores aceptados")
    a1.set_title("F3a. Se fija α y se despeja τ")
    a2.plot(t, tpr, color=GEN, lw=2)
    a2.axvline(tau, color=INK, lw=1, ls="--")
    a2.set_xlim(-0.1, 1.0)
    a2.set_ylim(-0.02, 1.02)
    a2.set_xlabel("Umbral τ")
    a2.set_ylabel("TPR(τ): genuinos aceptados")
    a2.set_title("F3b. Costo del umbral en rostros reales")
    fig.savefig(FIG / "F3-far-tpr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f4_bootstrap(plt, taus, tau, lo, hi):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(taus, bins=40, color=GEN, alpha=0.8)
    ax.axvline(tau, color=INK, lw=1.2, ls="--", label=f"τ* = {tau:.3f}".replace(".", ","))
    for v in (lo, hi):
        ax.axvline(v, color=INK2, lw=0.9, ls=":")
    ax.text(lo, ax.get_ylim()[1] * 0.7, f" {lo:.3f}".replace(".", ","), fontsize=8, color=INK2)
    ax.text(hi, ax.get_ylim()[1] * 0.7, f" {hi:.3f}".replace(".", ","), fontsize=8, color=INK2)
    ax.set_xlabel("τ* recalculado en cada réplica (remuestreo de identidades)")
    ax.set_ylabel(f"Réplicas (de {len(taus):,})".replace(",", "."))
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("F4. Incertidumbre de τ*: bootstrap por identidad, IC 95 %")
    fig.savefig(FIG / "F4-bootstrap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f5_matriz(plt, sc):
    ids = np.unique(np.concatenate([sc.imp_a, sc.gen_id]))
    pos = {g: k for k, g in enumerate(ids)}
    m = np.full((len(ids), len(ids)), np.nan)
    for a, p, s in zip(sc.imp_a, sc.imp_b, sc.impostor):
        i, j = pos[p], pos[a]
        m[i, j] = s if np.isnan(m[i, j]) else max(m[i, j], s)
    for g in ids:
        m[pos[g], pos[g]] = sc.genuine[sc.gen_id == g].mean()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("azul", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(False)
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
    ticks = [k for k, g in enumerate(ids) if g in (1, 10, 20, 32, 40, 48)]
    ax.set_xticks(ticks, [ids[k] for k in ticks])
    ax.set_yticks(ticks, [ids[k] for k in ticks])
    ax.set_xlabel("Ancla de la identidad")
    ax.set_ylabel("Fotos de la identidad")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Coseno (diagonal: media genuina; fuera: máximo impostor)")
    ax.set_title("F5. Cada identidad contra el ancla de cada identidad")
    fig.savefig(FIG / "F5-matriz.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f6_parejas(plt, fa):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labs = [f"{a}–{b}" for (a, b), _ in fa]
    vals = [n for _, n in fa]
    ax.barh(labs[::-1], vals[::-1], color=IMP, height=0.6)
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.6, y, str(v), va="center", fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Falsas aceptaciones a τ* (total {sum(vals)})")
    ax.set_ylabel("Pareja de identidades")
    ax.set_title("F6. De qué parejas de personas sale la cola impostora")
    fig.savefig(FIG / "F6-parejas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f7_margen(plt, sc, tau):
    ids = np.unique(sc.gen_id)
    gmin = np.array([sc.genuine[sc.gen_id == g].min() for g in ids])
    imax = np.array([sc.impostor[(sc.imp_b == g)].max() for g in ids])
    order = np.argsort(gmin)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(ids))
    ax.vlines(x, imax[order], gmin[order], color=GRID, lw=1.5)
    ax.scatter(x, gmin[order], s=22, color=GEN, label="Genuino más bajo de la identidad", zorder=3)
    ax.scatter(x, imax[order], s=22, color=IMP, label="Impostor más alto (sus fotos contra otras anclas)", zorder=3)
    ax.axhline(tau, color=INK, lw=1, ls="--")
    ax.text(len(ids) - 0.5, tau + 0.015, f"τ* = {tau:.3f}".replace(".", ","), ha="right", fontsize=8.5)
    ax.set_xticks(x, ids[order], fontsize=6.5, rotation=90)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Identidad (ordenada por su genuino más bajo)")
    ax.set_ylabel("Coseno con el ancla")
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("F7. Margen por identidad: ningún genuino cae bajo τ*")
    fig.savefig(FIG / "F7-margen.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {int(g): [float(a), float(b)] for g, a, b in zip(ids, gmin, imax)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt = style()
    sc, d = scores_and_ids()
    ref = json.loads((CALIBRACION / "resultado.json").read_text(encoding="utf-8"))["puntos"]
    tau, tau4 = ref[0]["tau"], ref[1]["tau"]
    lo, hi = ref[0]["tau_ic95"]
    taus = np.load(CALIBRACION / "bootstrap_tau.npz")["1e-03"]
    fa = false_accepts_by_pair(sc, tau)
    f1_particion(plt)
    f2_histograma(plt, sc, tau)
    f3_far_tpr(plt, sc, tau, tau4)
    f4_bootstrap(plt, taus, tau, lo, hi)
    f5_matriz(plt, sc)
    f6_parejas(plt, fa)
    margen = f7_margen(plt, sc, tau)
    q = [1, 5, 50, 95, 99, 99.9]
    cifras = {
        "fotos_usadas": int(len(d["photo"])), "identidades": int(len(np.unique(d["identity"]))),
        "genuinos": int(len(sc.genuine)), "impostores": int(len(sc.impostor)),
        "impostor": {"media": float(sc.impostor.mean()), "de": float(sc.impostor.std()),
                     "max": float(sc.impostor.max()), **{f"p{p}": float(np.percentile(sc.impostor, p)) for p in q}},
        "genuino": {"media": float(sc.genuine.mean()), "de": float(sc.genuine.std()),
                    "min": float(sc.genuine.min()), **{f"p{p}": float(np.percentile(sc.genuine, p)) for p in q}},
        "tau": tau, "tau_ic95": [lo, hi],
        "falsas_aceptaciones_por_pareja": [[list(map(int, p)), n] for p, n in fa],
        "identidad_menor_margen": min(margen.items(), key=lambda kv: kv[1][0] - kv[1][1]),
    }
    (FIG / "cifras.json").write_text(json.dumps(cifras, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in cifras.items() if k != "falsas_aceptaciones_por_pareja"}, indent=1))


if __name__ == "__main__":
    main()
