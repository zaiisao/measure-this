#!/usr/bin/env python3
"""Extract onset evidence for the selected performances (all test performances; one performance
per train/val work, the first in sorted order). Writes results/onsets/<stem>.npz and checks the
audio/annotation alignment (median lag of envelope peaks around annotated beats)."""
import csv
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from subdivision_hmm import onsets as O
from subdivision_hmm.config import INVENTORY as INV

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/onsets"


def selected():
    perfs = defaultdict(set)
    for line in (HERE / "results/intervals.jsonl").open():
        x = json.loads(line)
        perfs[(x["split"], x["work_id"])].add((x["performance_id"], x["feature_stem"]))
    out = []
    for (split, work), ps in sorted(perfs.items()):
        out.extend(sorted(ps) if split == "test" else sorted(ps)[:1])
    return out


def lag_check(env, fps, beats):
    """Median offset (s) of the envelope maximum within +-100 ms of each annotated beat."""
    lags = []
    for b in beats:
        i = int(round(b * fps)); w = int(0.1 * fps)
        if i - w < 0 or i + w >= len(env):
            continue
        lags.append((np.argmax(env[i - w:i + w + 1]) - w) / fps)
    return float(np.median(lags)) if lags else float("nan")


def job(args):
    perf_id, stem, member, start, end, beats = args
    path = OUT / f"{stem}.npz"
    if path.exists():
        return stem, "cached"
    try:
        ea, fa, ta = O.audio100(member, start, end)
        try:
            em, fm, tm = O.mel50(stem)
        except KeyError:  # performance absent from the Explore This feature store
            em, fm, tm = np.zeros(0, np.float32), 50.0, np.zeros(0)
        tp = O.perfmidi(perf_id)
        np.savez_compressed(path, env_audio100=ea, fps_audio100=fa, on_audio100=ta,
                            env_mel50=em, fps_mel50=fm, on_mel50=tm, on_perfmidi=tp,
                            lag_audio100=lag_check(ea, fa, beats), lag_mel50=lag_check(em, fm, beats))
        return stem, "ok"
    except Exception as e:  # report, do not hide
        return stem, f"error: {e!r}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = {r["performance_id"]: r for r in csv.DictReader(INV.open())}
    meta = {r["midi_performance"]: r for r in csv.DictReader((O.ASAP / "metadata.csv").open())}
    ann = json.loads((O.ASAP / "asap_annotations.json").read_text())
    jobs = []
    for perf_id, stem in selected():
        m = meta[perf_id]
        beats = json.loads(str(ann[perf_id]["performance_beats"]).replace("'", '"')) if isinstance(ann[perf_id]["performance_beats"], str) else ann[perf_id]["performance_beats"]
        jobs.append((perf_id, stem, inv[perf_id]["maestro_audio_member"], float(m["start"] or 0),
                     float(m["end"]) if m["end"] else None, beats))
    print(f"{len(jobs)} performances", flush=True)
    with Pool(12) as pool:
        for i, (stem, status) in enumerate(pool.imap_unordered(job, jobs)):
            if status != "ok" or (i + 1) % 20 == 0:
                print(i + 1, stem, status, flush=True)


if __name__ == "__main__":
    main()
