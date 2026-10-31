#!/usr/bin/env python3
"""Run a JSONL file of RunSpec jobs in parallel (CPU, 2 threads per job); skip finished jobs.

Each job trains, then evaluates the checkpoints chosen by (a) the original policy
(best validation weighted objective), (b) best validation raw BAcc, and (c) the final step.
"""
import argparse
import json
import os
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
from neural_subdivision_hmm.data_scaling import common  # noqa: E402


def run(job):
    spec = common.RunSpec(**job)
    out = common.RUNS / spec.name
    done = out / "summary.json"
    if done.exists():
        return spec.name, "cached"
    try:
        meta = common.train(spec)
        evals = {}
        for tag, ck in (("val_objective", "best_val_objective.pt"), ("val_bacc", "best_val_bacc.pt"), ("final", "final.pt")):
            if (out / ck).exists():
                evals[tag] = common.evaluate_checkpoint(out / ck, tag, out)
        done.write_text(json.dumps({"name": spec.name, "tags": spec.tags, "n_works": len(spec.work_ids),
                                    "seed": spec.seed, "budget": spec.budget, "optimizer_steps": meta["optimizer_steps"],
                                    "epochs_equivalent": meta["epochs_equivalent"],
                                    "best_val_objective_step": meta["best_val_objective"]["step"],
                                    "best_val_bacc_step": meta["best_val_bacc"]["step"],
                                    "wall_seconds": meta["wall_seconds"], "evals": evals}, indent=1))
        return spec.name, "ok"
    except Exception:
        return spec.name, "error:\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in a.jobs.read_text().splitlines() if l.strip()]
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for name, status in pool.imap_unordered(run, jobs):
            print(name, status, flush=True)


if __name__ == "__main__":
    main()
