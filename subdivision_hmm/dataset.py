"""Labelled beat-interval sequences and per-interval onset evidence (shared by both HMMs)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from subdivision_hmm.observation import resample_interval, standardise
from subdivision_hmm.onsets import envelope_from_events

HERE = Path(__file__).resolve().parent


def load_sequences():
    perfs = defaultdict(list)
    with (HERE / "results/intervals.jsonl").open() as intervals:
        for line in intervals:
            x = json.loads(line)
            if (HERE / f"results/onsets/{x['feature_stem']}.npz").exists():
                perfs[x["performance_id"]].append(x)
    seqs = []
    for pid, rows in perfs.items():
        rows.sort(key=lambda r: r["beat_index"])
        cur = [rows[0]]
        for r in rows[1:]:
            if r["beat_index"] == cur[-1]["beat_index"] + 1:
                cur.append(r)
            else:
                seqs.append(cur); cur = [r]
        seqs.append(cur)
    return [s for s in seqs if len(s) >= 2]


class Evidence:
    """Per-performance onset arrays, loaded once."""

    def __init__(self):
        self.cache = {}

    def get(self, stem):
        if stem not in self.cache:
            z = np.load(HERE / f"results/onsets/{stem}.npz")
            d = {k: z[k] for k in z.files}
            on = d["on_perfmidi"]
            d["env_perfmidi"] = envelope_from_events(on, 100.0, duration=on.max() + 2 if len(on) else 1)
            d["fps_perfmidi"] = 100.0
            for src in ("audio100", "mel50", "perfmidi"):
                if len(d[f"env_{src}"]):
                    d[f"std_{src}"] = standardise(d[f"env_{src}"])
            self.cache[stem] = d
        return self.cache[stem]


def interval_inputs(seqs, ev, src, raw=False):
    """Onset lists and resampled envelopes for every interval: standardised by default, the raw
    non-negative curve with raw=True (needed by the naive slot sum)."""
    ons, ys, ok = [], [], []
    for s in seqs:
        d = ev.get(s[0]["feature_stem"])
        has = f"std_{src}" in d
        for r in s:
            ons.append((d[f"on_{src}"], r["perf_start"], r["perf_end"]))
            key = f"env_{src}" if raw else f"std_{src}"
            ys.append(resample_interval(d[key], float(d[f"fps_{src}"]), r["perf_start"], r["perf_end"]) if has else np.zeros(48))
            ok.append(has)
    return ons, np.stack(ys), np.array(ok)
