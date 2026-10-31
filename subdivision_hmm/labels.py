"""Score-derived notated-subdivision labels for ASAP performance beat intervals.

Definition (notated rhythmic subdivision, measure level)
-------------------------------------------------------
For a score beat interval [s_i, s_{i+1}) (ASAP's annotated beat, i.e. the beat unit ASAP marks
for that score; not re-derived from the time signature), every score-MIDI note attack strictly
inside the interval has a beat phase x = (t - s_i) / (s_{i+1} - s_i). Attacks within TOL of 0 or 1
are beat attacks and carry no subdivision information.

A ratio r *fits* an attack if min_k |x - k/r| <= TOL. The label of a score measure (span between
ASAP score downbeats) is the smallest r in FULL_RATIOS that fits every interior attack of every
beat interval in the measure, i.e. the coarsest regular lattice that contains all notated attacks
of the measure. Every beat of the measure receives this label, also beats that themselves contain
only a long note: the label describes the notated rhythmic grid of the measure, not onset density.

* Tuplets enter through their written onset positions (triplet eighths -> phases 1/3, 2/3).
* Ties and rests produce no attack, so they do not contribute (the label is the attack lattice).
* Grace notes, arpeggiation offsets, quintuplets, etc. produce attacks off every lattice in
  FULL_RATIOS: the measure label becomes UNKNOWN (no label is forced).
* A measure needs >= 2 eligible beat intervals; intervals touching ASAP 'bR' beats are dropped.

Confidence: HIGH if the track's score/performance match is good (>= 90% notes matched, p95
residual <= 60 ms; data/performance_inventory.csv) and the label is exercised repeatedly:
>= 2 beats of the measure carry an attack at a phase diagnostic of the label (on the label lattice
but on no coarser lattice in FULL_RATIOS that divides it); for label 1, >= 2 beats with a beat
attack and no interior attack anywhere in the measure.

This is neither the perceptual tatum nor the performed onset density; see the report.
"""
import ast
import bisect
import json
from collections import defaultdict
from pathlib import Path

import mido

from subdivision_hmm.config import ASAP

FULL_RATIOS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
TOL = 0.015  # beat-phase tolerance in score time (same as the earlier reference builder)
UNKNOWN = 0


def fits(x, r, tol=TOL):
    return abs(x * r - round(x * r)) <= tol * r


def label_for(phases, ratios=FULL_RATIOS):
    for r in ratios:
        if all(fits(x, r) for x in phases):
            return r
    return UNKNOWN


def diagnostic(x, r):
    """Attack at phase x is on lattice r but on no coarser lattice dividing r."""
    if not fits(x, r):
        return False
    return not any(fits(x, d) for d in FULL_RATIOS if d < r and r % d == 0)


def note_onset_times(path):
    """Sorted note-on times (s) of a MIDI file, all tracks, tempo map applied."""
    t, out = 0.0, []
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            out.append(t)
    return sorted(out)


def _parse(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def performance_intervals(perf_id, ann, quality_ok):
    """Labelled beat intervals of one ASAP performance (performance-time beats)."""
    sb = list(map(float, _parse(ann["midi_score_beats"])))
    pb = list(map(float, _parse(ann["performance_beats"])))
    if len(sb) != len(pb) or str(ann["score_and_performance_aligned"]) != "True":
        return []
    st = {round(float(k), 6): v for k, v in _parse(ann["midi_score_beats_type"]).items()}
    pt = {round(float(k), 6): v for k, v in _parse(ann["performance_beats_type"]).items()}
    downs = sorted(map(float, _parse(ann["midi_score_downbeats"])))
    ts = sorted((float(k), v[0]) for k, v in _parse(ann["midi_score_time_signatures"]).items())
    times = note_onset_times(ASAP / Path(perf_id).parent / "midi_score.mid")
    rows = []
    for i in range(len(sb) - 1):
        s0, s1, p0, p1 = sb[i], sb[i + 1], pb[i], pb[i + 1]
        if s1 <= s0 or p1 <= p0 or "R" in str(st.get(round(s0, 6), "")) + str(st.get(round(s1, 6), "")) \
                + str(pt.get(round(p0, 6), "")) + str(pt.get(round(p1, 6), "")):
            continue
        lo, hi = bisect.bisect_left(times, s0 - 1e-6), bisect.bisect_left(times, s1 - 1e-6)
        ph = sorted({round((t - s0) / (s1 - s0), 6) for t in times[lo:hi]})
        interior = [x for x in ph if TOL < x < 1 - TOL]
        tsi = bisect.bisect_right([t for t, _ in ts], s0 + 1e-6) - 1
        rows.append({"performance_id": perf_id, "beat_index": i, "score_start": s0, "score_end": s1,
                     "perf_start": p0, "perf_end": p1, "measure": bisect.bisect_right(downs, s0 + 1e-6),
                     "time_signature": ts[tsi][1] if tsi >= 0 else "", "interior_phases": interior,
                     "beat_attack": bool(ph and ph[0] <= TOL), "beat_label": label_for(interior),
                     "quality_ok": quality_ok})
    by_measure = defaultdict(list)
    for r in rows:
        by_measure[r["measure"]].append(r)
    for m, rs in by_measure.items():
        allph = [x for r in rs for x in r["interior_phases"]]
        lab = label_for(allph) if len(rs) >= 2 else UNKNOWN
        if lab == UNKNOWN:
            evidence = 0
        elif lab == 1:
            evidence = sum(r["beat_attack"] for r in rs)
        else:
            evidence = sum(any(diagnostic(x, lab) for x in r["interior_phases"]) for r in rs)
        off = [x for x in allph if lab == UNKNOWN and not any(fits(x, q) for q in FULL_RATIOS)]
        for r in rs:
            r["label"] = lab
            r["measure_evidence_beats"] = evidence
            r["high_confidence"] = bool(lab != UNKNOWN and evidence >= 2 and r["quality_ok"])
            r["near_beat_offlattice_attacks"] = sum(1 for x in off if x < 0.1 or x > 0.9)
            r["offlattice_attacks"] = len(off)
    return rows


def load_annotations():
    return json.loads((ASAP / "asap_annotations.json").read_text())
