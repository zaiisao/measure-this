#!/usr/bin/env python3
"""Aggregate runs/*/summary.json into tables and figures (results/, figures/).

Primary metric: validation-decoded balanced accuracy (decoder chosen on validation) on
VALIDATION (protocol decisions) and TEST (continuity with earlier reports; already examined).
Checkpoint policies: val_objective (original), val_bacc, final.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
COLORS = ["#2a6fdb", "#d4731a", "#2e9d6b", "#8a4fd3", "#c43d5a"]


def load():
    rows = []
    for p in sorted((HERE / "runs").glob("*/summary.json")):
        s = json.loads(p.read_text())
        for pol, e in s["evals"].items():
            t = s["tags"]
            rows.append({"name": s["name"], "budget": t.get("budget"), "protocol": t.get("protocol"), "n": s["n_works"],
                         "exp": t.get("exp", "stage1"), "subset": s["name"].rsplit("_s", 1)[0].split("_", 1)[-1],
                         "init_seed": s["seed"], "policy": pol, "steps": s["optimizer_steps"],
                         "epochs_eq": s["epochs_equivalent"], "sel_step": s[f"best_{pol}_step"] if pol != "final" else s["optimizer_steps"],
                         "val_bacc": e["val_decoded_bacc"], "val_raw_bacc": e["val_raw"]["bacc"],
                         "test_bacc": e["test_decoded"]["bacc"], "test_raw_bacc": e["test_raw"]["bacc"],
                         "test_acc": e["test_decoded"]["accuracy"], "pair_2_4": e["test_decoded"]["pair_2_4"],
                         "pair_3_6": e["test_decoded"]["pair_3_6"], "family_auc": e["test_family_auc"],
                         "test_nll": e["test_marginal_nll"], "test_state_nll": e["test_raw_posterior_state_nll"],
                         "ece": e["test_decoded"]["ece"], "brier": e["test_decoded"]["brier"],
                         "transitions": e["test_transitions_per_100"],
                         **{f"recall_{k}": v for k, v in e["test_decoded"]["recall"].items()},
                         **{f"val_recall_{k}": v for k, v in e["val_raw"]["recall"].items()},
                         **{f"pred_{k}": v for k, v in e["test_pred_distribution"].items()},
                         "tags": json.dumps(t)})
    return pd.DataFrame(rows)


def agg(df, by, cols=("val_bacc", "test_bacc")):
    out = []
    for key, g in df.groupby(by):
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r["runs"] = len(g); r["subsets"] = g.subset.nunique()
        for c in cols:
            r[f"{c}_mean"] = g[c].mean(); r[f"{c}_sd"] = g[c].std(ddof=1) if len(g) > 1 else float("nan")
            r[f"{c}_min"] = g[c].min(); r[f"{c}_max"] = g[c].max()
        out.append(r)
    return pd.DataFrame(out)


def variance_split(df, n):
    """Between-subset (one init seed each) vs between-init (same subset) SD at size n."""
    g = df[(df.n == n)]
    seeds = g.groupby("subset").filter(lambda x: x.init_seed.nunique() > 1)
    within = seeds.groupby("subset")["test_bacc"].std(ddof=1).mean()
    within_v = seeds.groupby("subset")["val_bacc"].std(ddof=1).mean()
    between = g[(g.init_seed == 0) & (g.protocol == "independent")]
    return {"n": n, "init_sd_test": within, "init_sd_val": within_v,
            "subset_sd_test": between.test_bacc.std(ddof=1), "subset_sd_val": between.val_bacc.std(ddof=1),
            "subsets": between.subset.nunique(), "seeded_subsets": seeds.subset.nunique()}


def fig_scaling(df, path, policy="val_objective"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, budget, title in zip(axes, ("B1", "B2"), ("matched epochs (≤10, original)", "matched updates (1,200 steps)")):
        g = df[(df.budget == budget) & (df.policy == policy)]
        for i, (proto, label) in enumerate((("independent", "independent random subsets"), ("nested", "nested random orders"),
                                            ("original_order", "original deterministic order"))):
            h = g[(g.protocol == proto) | ((g.protocol == "all") & (proto != "original_order" or True))]
            h = h[(h.protocol == proto) | (h.protocol == "all")]
            pts = h.groupby("n").test_bacc
            x = np.array(sorted(h.n.unique()))
            m = pts.mean().reindex(x).to_numpy()
            ax.plot(x + (i - 1) * 1.2, m, color=COLORS[i], lw=2, marker="o", ms=7, mec="white", mew=1.5, label=label)
            for n in x:
                v = h[h.n == n].test_bacc.to_numpy()
                ax.scatter(np.full(len(v), n + (i - 1) * 1.2), v, s=10, color=COLORS[i], alpha=.35, lw=0)
        ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel("training works", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("test balanced accuracy (decoder from val)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Scaling with training works (checkpoint policy: {policy}); dots = individual runs", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_curves(names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, (label, pattern) in enumerate(names):
        runs = sorted((HERE / "runs").glob(pattern))
        for j, r in enumerate(runs):
            h = json.loads((r / "training.json").read_text())["history"]
            st = [x["step"] for x in h]
            kw = dict(color=COLORS[i], lw=1.4, alpha=.8, label=label if j == 0 else None)
            axes[0].plot(st, [x["val_raw_bacc"] for x in h], **kw)
            axes[1].plot(st, [x["val_acoustic_nll"] for x in h], **kw)
            axes[2].plot(st, [x["val_state_nll_per_label"] for x in h], **kw)
    for ax, t in zip(axes, ("validation raw BAcc", "validation acoustic NLL / interval", "validation posterior-state NLL / label")):
        ax.set_title(t, fontsize=9, color=INK); ax.set_xlabel("optimizer updates", color=MUTED); ax.grid(color=GRID, lw=.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[2].set_yscale("log"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    df = load()
    df.to_csv(RES / "all_runs.csv", index=False)
    s1 = df[df.exp == "stage1"]
    table = agg(s1, ["budget", "policy", "protocol", "n"])
    table.to_csv(RES / "stage1_scaling.csv", index=False)
    pd.DataFrame([variance_split(s1[(s1.budget == b) & (s1.policy == "val_objective")], 30) | {"budget": b}
                  for b in ("B1", "B2") if len(s1)]).to_csv(RES / "stage1_variance_n30.csv", index=False)
    for pol in ("val_objective", "val_bacc"):
        fig_scaling(s1, FIG / f"scaling_{pol}.png", pol)
    fig_curves([("original 30 works", "B2_orig_n30_s*"), ("all 96 works", "B2_all_n96_s*"),
                ("random 30 (draw 0)", "B2_indep_n30_d0_s*")], FIG / "learning_curves_B2.png",
               "Matched-update training (1,200 steps): validation curves")
    print(table[table.policy == "val_objective"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
