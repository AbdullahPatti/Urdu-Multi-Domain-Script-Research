"""Stage 6 — paper figures (vector PDF), one consistent style.

One accent hue only (the paper's link colour, dark blue #1f4e8c) plus neutral greys,
so the figures match the text and print well in greyscale: fine-tuned encoders are
blue, few-shot LLMs are grey, and marker shape identifies each model, so identity
never rests on colour alone. Baselines are dashed/dotted reference lines, heatmaps
use a single-hue blue ramp, and data origin uses blue/grey tints with a legend.
Chart text is a clean sans-serif.

Usage:  python -m pipeline.figures          (writes paper/figures/*.pdf)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from pipeline.analyze import baselines, length_matrix, load_runs, matrix
from pipeline.config import ROOT, TASK_BY_ID

OUT = ROOT / "paper" / "figures"
INK, INK2, GRID, MUTED = "#1a1a1a", "#555555", "#e6e6e6", "#a0a0a0"
ACCENT = "#1f4e8c"
FAMILY_COLOR = {"ft": ACCENT, "llm": "#7a7a7a"}
ORIGIN_COLOR = {"organic": ACCENT, "generated": "#8fb0d9", "translated": "#6b6b6b", "uncertain": "#c4c4c4"}
MODELS = [("ft", "XLM-R", "o"), ("ft", "mBERT", "s"),
          ("llm", "Llama-3.1-8B", "^"), ("llm", "Qwen2.5-7B", "D"), ("llm", "Mistral-7B", "p")]
TITLES = {"N-SA": "Nastaliq · Sentiment", "N-HS": "Nastaliq · Hate speech", "N-FN": "Nastaliq · Fake news",
          "N-QA": "Nastaliq · QA validation", "R-SA": "Roman · Sentiment",
          "R-CA": "Roman · Cyber abuse", "R-P3": "Roman · 3-class polarity"}
ORDER = ["N-SA", "N-HS", "N-FN", "N-QA", "R-SA", "R-CA", "R-P3"]
# sequential blue ramp (reference palette steps 100..700, lightest = near zero)
GREYS = LinearSegmentedColormap.from_list(
    "blues", ["#ffffff", "#dbe5f2", "#b3c8e3", "#7f9fcb", "#4a73ab", "#1f4e8c", "#143766"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def mean_matrix(df, tid, fam, model):
    g = df[(df.task == tid) & (df.family == fam) & (df.model == model)]
    if g.empty:
        return None
    mats = [matrix(gs, TASK_BY_ID[tid]) for _, gs in g.groupby("seed")]
    m = sum(mats) / len(mats)
    return None if m.isna().values.any() else m


def available(df):
    return [t for t in ORDER if (df.task == t).any()]


# --------------------------------------------------------------------------- figure: SS vs ST
def fig_ss_st(df):
    tasks = available(df)
    n = len(tasks)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.05 * rows), sharex=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, tid in zip(axes.flat, tasks):
        task = TASK_BY_ID[tid]
        _, rnd = baselines(task)
        lm = length_matrix(task)
        len_st = np.nanmean([lm.iloc[i, j] for i in range(len(lm)) for j in range(len(lm)) if i != j])
        ylabels = []
        for y, (fam, model, mk) in enumerate(MODELS):
            m = mean_matrix(df, tid, fam, model)
            ylabels.append(model)
            if m is None:
                continue
            ss = np.mean(np.diag(m.values))
            st = np.nanmean(m.values[~np.eye(len(m), dtype=bool)])
            c = FAMILY_COLOR[fam]
            ax.plot([st, ss], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
            ax.scatter([ss], [y], marker=mk, s=26, facecolor="white", edgecolor=c, lw=1.1, zorder=3)
            ax.scatter([st], [y], marker=mk, s=26, color=c, edgecolor="white", lw=0.6, zorder=4)
        ax.axvline(rnd, color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.axvline(len_st, color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        ax.set_yticks(range(len(MODELS)), ylabels)
        ax.invert_yaxis()
        ax.set_xlim(0.2, 1.0)
        ax.set_title(TITLES[tid], loc="left", pad=3)
        ax.grid(axis="y", visible=False)
        if ax not in axes[:, 0]:
            ax.set_yticklabels([])
    for ax in axes[-1]:
        ax.set_xlabel("Macro-F1")
    # legend: open = in-domain (SS), filled = cross-domain (ST); line styles = baselines
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="In-domain (SS)"),
               Line2D([], [], marker="o", ls="", color=INK2, label="Cross-domain (ST)"),
               Line2D([], [], color=FAMILY_COLOR["ft"], lw=1.4, label="Fine-tuned"),
               Line2D([], [], color=FAMILY_COLOR["llm"], lw=1.4, label="Few-shot LLM"),
               Line2D([], [], color=INK2, lw=0.7, ls=(0, (4, 2)), label="Uniform random"),
               Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.5)), label="Length-only (ST)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "ss_st.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- figure: transfer matrices
def fig_matrices(df, script):
    tasks = [t for t in available(df) if TASK_BY_ID[t].script == script]
    models = [(f, m) for f, m, _ in MODELS]
    if not tasks:
        return
    fig, axes = plt.subplots(len(tasks), len(models), figsize=(7.0, 1.55 * len(tasks) + 0.3), squeeze=False)
    for r, tid in enumerate(tasks):
        for c, (fam, model) in enumerate(models):
            ax = axes[r, c]
            ax.grid(False)
            m = mean_matrix(df, tid, fam, model)
            if r == 0:
                ax.set_title(model, pad=3)
            if m is None:
                ax.axis("off")
                continue
            ax.imshow(m.values, cmap=GREYS, vmin=0.2, vmax=1.0)
            n = len(m)
            fs = 5.2 if n > 4 else 6
            for i in range(n):
                for j in range(n):
                    v = m.values[i, j]
                    ax.text(j, i, f"{v:.2f}"[1:] if round(v, 2) < 1 else "1.0", ha="center", va="center", fontsize=fs,
                            color="white" if v > 0.66 else INK, fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(n), [s.split("-")[-1] if len({x.split('-')[0] for x in m.columns}) == 1 else s
                                     for s in m.columns], fontsize=5.5)
            ax.set_yticks(range(n), [s.split("-")[-1] if len({x.split('-')[0] for x in m.index}) == 1 else s
                                     for s in m.index], fontsize=5.5)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(TITLES[tid].split(" · ")[1] + "\nsource", fontsize=6.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(OUT / f"matrices_{script.lower()}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- figure: source vs target drop
def fig_drops(df):
    tasks = available(df)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True, sharey=True)
    for ax, fam, title in zip(axes, ("ft", "llm"), ("Fine-tuned encoders", "Few-shot LLMs")):
        pts = []
        for tid in tasks:
            for f, model, mk in MODELS:
                if f != fam:
                    continue
                m = mean_matrix(df, tid, f, model)
                if m is None:
                    continue
                ids = list(m.index)
                for i in ids:
                    for j in ids:
                        if i != j:
                            pts.append((m.loc[i, i] - m.loc[i, j], m.loc[j, j] - m.loc[i, j], mk))
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.fill_between([0, 1], -1, 0, color="#f0efec", zorder=0)
        for mk in {p[2] for p in pts}:
            xs = [p[0] for p in pts if p[2] == mk]
            ys = [p[1] for p in pts if p[2] == mk]
            ax.scatter(xs, ys, marker=mk, s=14, facecolor=FAMILY_COLOR[fam], alpha=0.75,
                       edgecolor="white", lw=0.4, label=next(mm for ff, mm, k in MODELS if k == mk))
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel(r"Source Drop $\Delta_S$ (F1)")
        ax.text(0.97, 0.03, "harder target\n" + r"$\Delta_S>0,\ \Delta_T\leq 0$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color=INK2)
        ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    axes[0].set_ylabel(r"Target Drop $\Delta_T$ (F1)")
    lim = (-0.4, 0.85)
    axes[0].set_xlim(*lim)
    axes[0].set_ylim(*lim)
    fig.tight_layout()
    fig.savefig(OUT / "source_vs_target_drop.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- figure: benchmark overview
def fig_overview():
    """Size of every domain (train + test, log scale), grouped by transfer matrix and
    coloured by data origin."""
    from matplotlib.patches import Patch
    from pipeline.config import ANALYSIS, DOMAIN_BY_KEY
    audit = pd.read_csv(ANALYSIS / "data_audit_v11.csv", encoding="utf-8-sig").set_index("key")
    rows = []
    for tid in ORDER:
        for k in TASK_BY_ID[tid].domains:
            d = DOMAIN_BY_KEY[k]
            rows.append((tid, d.sid, int(audit.loc[k, "v1.1_train"] + audit.loc[k, "v1.1_test"]), d.origin))
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    x, ticks, labels, centers = 0, [], [], []
    for tid in ORDER:
        grp = [r for r in rows if r[0] == tid]
        start = x
        for _, sid, n, origin in grp:
            ax.bar(x, n, width=0.78, color=ORIGIN_COLOR[origin], edgecolor="white", linewidth=0.8)
            ticks.append(x); labels.append(sid.split("-")[1] if tid != "R-P3" else sid.replace("-", "\n"))
            x += 1
        centers.append(((start + x - 1) / 2, TITLES[tid].replace(" · ", "\n")))
        x += 0.8
    ax.set_yscale("log")
    ax.set_ylim(80, 60000)
    ax.set_xticks(ticks, labels, fontsize=6)
    ax.set_ylabel("Examples (train + test, log scale)")
    ax.grid(axis="x", visible=False)
    for cx, name in centers:
        ax.text(cx, -0.2, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color=INK2)
    handles = [Patch(color=c, label=o) for o, c in ORIGIN_COLOR.items()]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.1))
    fig.tight_layout()
    fig.savefig(OUT / "overview.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- figure: bootstrap forest plot
def fig_forest():
    """Fine-tuned minus few-shot mean ST with 95% bootstrap CIs for every pair.
    Filled marker = significant after Holm correction; open = not significant."""
    from pipeline.config import ANALYSIS
    bt = pd.read_csv(ANALYSIS / "bootstrap_ft_vs_llm.csv")
    llm_marker = {"Llama-3.1-8B": "^", "Qwen2.5-7B": "D", "Mistral-7B": "p"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    for ax, ft in zip(axes, ("XLM-R", "mBERT")):
        y, yt, yl = 0, [], []
        for tid in ORDER:
            g = bt[(bt.task == tid) & (bt.a == ft)]
            for off, (_, r) in zip((-0.22, 0, 0.22), g.iterrows()):
                sig = r.p_holm < 0.05
                col = FAMILY_COLOR["ft"] if r.diff_ST > 0 and sig else FAMILY_COLOR["llm"] if sig else MUTED
                ax.plot([100 * r.ci_lo, 100 * r.ci_hi], [y + off] * 2, color=col, lw=1.2, solid_capstyle="round")
                ax.scatter(100 * r.diff_ST, y + off, marker=llm_marker[r.b], s=20, zorder=3,
                           facecolor=col if sig else "white", edgecolor=col, lw=1.0)
            yt.append(y); yl.append(TITLES[tid])
            y += 1
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_yticks(yt, yl)
        if ft == "XLM-R":
            ax.invert_yaxis()   # shared y: invert once
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{ft} minus few-shot LLM", loc="left", pad=3)
        ax.set_xlabel("Difference in cross-domain macro-F1 (pp)")
        ax.text(0.02, 0.01, "LLM better", transform=ax.transAxes, fontsize=6.5, color=FAMILY_COLOR["llm"])
        ax.text(0.98, 0.01, "fine-tuned better", transform=ax.transAxes, ha="right", fontsize=6.5,
                color=FAMILY_COLOR["ft"])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, ls="", mfc="white", mec=INK2, label=n) for n, m in llm_marker.items()]
    handles.append(Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, label="significant (Holm)"))
    handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, label="not significant"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "forest.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- figure: confusion, worst shifts
def fig_confusion():
    """Row-normalised confusion matrices (summed over 3 seeds) for the worst-transferring
    shift of each matrix, from analysis/errors_confusion.csv (pipeline.errors)."""
    from pipeline.config import ANALYSIS
    cf = pd.read_csv(ANALYSIS / "errors_confusion.csv")
    tasks = [t for t in ORDER if (cf.task == t).any()]
    cols = [("ft_tgt", "XLM-R, trained on target"), ("ft_src", "XLM-R, trained on source"),
            ("llm_src", "Top-ST LLM, source demos")]
    fig, axes = plt.subplots(len(tasks), 3, figsize=(5.4, 1.28 * len(tasks) + 0.3), squeeze=False)
    for r, tid in enumerate(tasks):
        for c, (tag, title) in enumerate(cols):
            ax = axes[r, c]
            ax.grid(False)
            g = cf[(cf.task == tid) & (cf.system == tag)]
            labels = list(dict.fromkeys(g["true"]))
            preds = labels + (["unparseable"] if g[g.pred == "unparseable"]["count"].sum() > 0 else [])
            M = np.array([[g[(g["true"] == a) & (g.pred == b)]["count"].sum() for b in preds] for a in labels], float)
            N = M / M.sum(1, keepdims=True)
            ax.imshow(N, cmap=GREYS, vmin=0, vmax=1, aspect="auto")
            for i in range(N.shape[0]):
                for j in range(N.shape[1]):
                    ax.text(j, i, f"{100 * N[i, j]:.0f}", ha="center", va="center", fontsize=5.8,
                            color="white" if N[i, j] > 0.55 else INK)
            short = [x[:3] if x != "unparseable" else "n/p" for x in preds]
            ax.set_xticks(range(len(preds)), short, fontsize=5.5)
            ax.set_yticks(range(len(labels)), [x[:3] for x in labels], fontsize=5.5)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            s0 = g.iloc[0]
            shift = f"{s0.src}→{s0.tgt}" if tag != "ft_tgt" else f"{s0.tgt}→{s0.tgt}"
            if tag == "llm_src":
                shift += f"\n{s0.model}"
            ax.set_title(shift, fontsize=6, pad=2)
            if r == 0:
                ax.text(0.5, 1.55, title, transform=ax.transAxes,
                        ha="center", fontsize=6.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(TITLES[tid].replace(" · ", "\n") + "\ntrue", fontsize=6)
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    fig.savefig(OUT / "confusion_worst.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_overview()
    fig_forest()
    fig_confusion()
    df = load_runs()
    fig_ss_st(df)
    fig_matrices(df, "Nastaliq")
    fig_matrices(df, "Roman")
    fig_drops(df)
    print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
