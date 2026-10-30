#!/usr/bin/env python3
"""Tables and figures from results/experiment: main table, ablations, confusion matrices,
oracle-vs-inference diagnostic, per-work / per-composer breakdowns. Writes results/analysis.md."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
EXP = HERE / "results/experiment"
FIG = HERE / "figures"
R = [1, 2, 3, 4, 6]
INK, MUTED, GRID, SURFACE = "#1f2328", "#59636e", "#d0d7de", "#ffffff"
SERIES = {"audio100": "#2a6fdb", "mel50": "#d4731a", "perfmidi": "#2e9d6b"}  # fixed order: blue, orange, green
SRC_NAME = {"audio100": "audio onsets", "mel50": "Explore This mel flux", "perfmidi": "performance MIDI (ideal)"}


def rows():
    out = [r for r in csv.DictReader((EXP / "results.csv").open()) if r["model"] != "naive"]
    if (EXP / "results_naive_raw.csv").exists():
        out += list(csv.DictReader((EXP / "results_naive_raw.csv").open()))
    return out


def f(v, d=3):
    try:
        x = float(v)
        return "–" if not np.isfinite(x) else f"{x:.{d}f}"
    except (TypeError, ValueError):
        return "–"


def main_tables(rs):
    L = []
    for subset in ("all_labelled", "high_confidence"):
        L += [f"\n### Test, vocabulary {{1,2,3,4,6}}, {subset.replace('_', ' ')}\n",
              "| Onset source | Likelihood | Decoder | κ | p_stay | Bal. acc | Acc | 1× | 2× | 3× | 4× | 6× | Binary (2,4) | Ternary (3,6) | Pred / ref transitions per 100 beats | Mean conf. | ECE |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        sel = [r for r in rs if r["vocab"] == "1-2-3-4-6" and r["split"] == "test" and r["subset"] == subset and r["decoder"] != "ablation_p_stay"]
        order = {"prior_constant": 0, "independent": 1, "hmm_uniform": 2, "hmm": 3}
        for r in sorted(sel, key=lambda r: (r["source"], r["model"], order[r["decoder"]])):
            L.append(f"| {r['source']} | {r['model']} | {r['decoder']} | {r['kappa'] or '–'} | {r['p_stay'] or '–'} | **{f(r['balanced_accuracy'])}** | {f(r['accuracy'])} | "
                     + " | ".join(f(r.get(f"acc_{k}")) for k in R)
                     + f" | {f(r['acc_binary'])} | {f(r['acc_ternary'])} | {f(r['pred_transitions_per_100_beats'],1)} / {f(r['ref_transitions_per_100_beats'],1)} | {f(r['mean_confidence'])} | {f(r['ece'])} |")
    return L


def vocab_table(rs):
    L = ["\n### Vocabulary sensitivity (test, all labelled, HMM with val-selected κ, p_stay)\n",
         "| Onset source | Likelihood | Vocabulary | In-vocabulary coverage | Bal. acc | Acc |", "|---|---|---|---:|---:|---:|"]
    for r in sorted([r for r in rs if r["split"] == "test" and r["subset"] == "all_labelled" and r["decoder"] == "hmm"],
                    key=lambda r: (r["source"], r["model"], r["vocab"])):
        L.append(f"| {r['source']} | {r['model']} | {{{r['vocab'].replace('-', ',')}}} | {f(r['coverage_in_vocab'])} | {f(r['balanced_accuracy'])} | {f(r['accuracy'])} |")
    return L


def p_stay_ablation(rs):
    L = ["\n### Transition-strength ablation (test, vocabulary {1,2,3,4,6}, κ fixed at the val choice)\n",
         "| Onset source | Likelihood | p_stay | Bal. acc | Pred transitions / 100 beats |", "|---|---|---:|---:|---:|"]
    sel = [r for r in rs if r["decoder"] == "ablation_p_stay" and r["vocab"] == "1-2-3-4-6"]
    for r in sorted(sel, key=lambda r: (r["source"], r["model"], float(r["p_stay"]))):
        L.append(f"| {r['source']} | {r['model']} | {float(r['p_stay']):.3g} | {f(r['balanced_accuracy'])} | {f(r['pred_transitions_per_100_beats'],1)} |")
    return L, sel


def preds(src, model, dec):
    p = EXP / f"pred_{src}_{model}_{dec}.csv"
    return list(csv.DictReader(p.open())) if p.exists() else []


def confusion(ps):
    m = np.zeros((5, 5), int)
    for r in ps:
        y = int(r["label"])
        if y in R:
            m[R.index(y), R.index(int(r["pred"]))] += 1
    return m


def diagnostics(src, model):
    """Errors of the HMM split into: emission prefers the reference (inference/prior caused the
    error) vs emission prefers the prediction (observation failure). Also oracle coverage."""
    ps = preds(src, model, "hmm")
    lab = [r for r in ps if int(r["label"]) > 0]
    inv = [r for r in lab if int(r["label"]) in R]
    err = [r for r in inv if int(r["pred"]) != int(r["label"])]
    emis_ref = sum(float(r[f"ll_{r['label']}"]) > float(r[f"ll_{r['pred']}"]) for r in err)
    indep = {(r["performance_id"], r["beat_index"]): r for r in preds(src, model, "independent")}
    fixed = sum(1 for r in inv if int(r["pred"]) == int(r["label"]) and int(indep[(r["performance_id"], r["beat_index"])]["pred"]) != int(r["label"]))
    broke = sum(1 for r in inv if int(r["pred"]) != int(r["label"]) and int(indep[(r["performance_id"], r["beat_index"])]["pred"]) == int(r["label"]))
    return {"labelled": len(lab), "oracle_coverage": len(inv) / max(1, len(lab)), "errors": len(err),
            "errors_emission_prefers_reference": emis_ref, "errors_emission_prefers_prediction": len(err) - emis_ref,
            "hmm_fixes_vs_independent": fixed, "hmm_breaks_vs_independent": broke}


def breakdown(ps, key):
    acc = defaultdict(lambda: defaultdict(list))
    for r in ps:
        y = int(r["label"])
        if y in R:
            acc[r[key]][y].append(int(r["pred"]) == y)
    out = []
    for k, d in sorted(acc.items()):
        out.append((k, sum(len(v) for v in d.values()), float(np.mean([np.mean(v) for v in d.values()])),
                    {c: float(np.mean(v)) for c, v in d.items()}))
    return out


def plot_confusions(cms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4.2), facecolor=SURFACE)
    for ax, (title, m) in zip(np.atleast_1d(axes), cms):
        norm = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color=SURFACE if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), [f"{r}×" for r in R]); ax.set_yticks(range(5), [f"{r}×" for r in R])
        ax.set_xlabel("predicted", color=MUTED); ax.set_ylabel("score-derived reference", color=MUTED)
        ax.set_title(title, fontsize=10, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pstay(sel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for src in SERIES:
        pts = sorted((float(r["p_stay"]), float(r["balanced_accuracy"])) for r in sel if r["source"] == src and r["model"] == "poisson")
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=SERIES[src], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=SRC_NAME[src])
        ax.annotate(SRC_NAME[src], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    ax.set_xscale("logit"); ax.set_xlabel("self-transition probability p_stay (0.2 = uniform)", color=MUTED)
    ax.set_ylabel("test balanced accuracy", color=MUTED); ax.axhline(0.2, color=GRID, lw=1, ls="--")
    ax.text(0.9995, 0.205, "chance", ha="right", color=MUTED, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.62))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Poisson occupancy likelihood: effect of persistence", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    rs = rows()
    L = ["# Analysis tables (generated)"] + main_tables(rs) + vocab_table(rs)
    pl, sel = p_stay_ablation(rs)
    L += pl
    L += ["\n### Oracle vs inference diagnostic (test, HMM, vocabulary {1,2,3,4,6})\n",
          "| Onset source | Likelihood | Oracle coverage | Errors | Emission prefers reference (inference error) | Emission prefers prediction (observation error) | HMM fixes vs independent | HMM breaks vs independent |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    cms = []
    for src in SERIES:
        for model in ("poisson", "gauss", "naive"):
            if not preds(src, model, "hmm"):
                continue
            d = diagnostics(src, model)
            L.append(f"| {src} | {model} | {d['oracle_coverage']:.3f} | {d['errors']} | {d['errors_emission_prefers_reference']} | "
                     f"{d['errors_emission_prefers_prediction']} | {d['hmm_fixes_vs_independent']} | {d['hmm_breaks_vs_independent']} |")
            m = confusion(preds(src, model, "hmm"))
            np.savetxt(HERE / f"results/confusion_{src}_{model}_hmm.csv", m, fmt="%d", delimiter=",",
                       header="rows=reference " + ",".join(map(str, R)) + "; cols=predicted", comments="")
            if model == "poisson":
                cms.append((f"{SRC_NAME[src]}, Poisson + HMM", m))
    if cms:
        plot_confusions(cms, FIG / "confusion_poisson_hmm.png")
    plot_pstay(sel, FIG / "p_stay_ablation.png")
    best = preds("audio100", "poisson", "hmm")
    if best:
        for key, title in (("work_id", "work"), ("composer", "composer (texture proxy)")):
            L += [f"\n### Per-{title}: audio onsets, Poisson + HMM (test)\n", f"| {title} | Labelled beats | Bal. acc | " + " | ".join(f"{r}×" for r in R) + " |",
                  "|---|---:|---:|" + "---:|" * len(R)]
            for k, n, ba, per in breakdown(best, key):
                L.append(f"| {k} | {n} | {ba:.3f} | " + " | ".join(f(per.get(r)) for r in R) + " |")
    (HERE / "results/analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
