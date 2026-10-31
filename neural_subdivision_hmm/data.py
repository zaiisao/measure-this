"""Adapter for the frozen subdivision_hmm interval and mel50 artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from subdivision_hmm import dataset as classical

RATIOS = (1, 2, 3, 4, 6)
K = len(RATIOS)


@dataclass
class BeatSequence:
    work_id: str
    performance_id: str
    stem: str
    split: str
    beat_index: np.ndarray
    y: np.ndarray
    log_duration: np.ndarray
    observed: np.ndarray
    labels: np.ndarray
    exact_allowed: np.ndarray
    family_allowed: np.ndarray
    high_confidence: np.ndarray

    @property
    def n(self):
        return len(self.labels)

    @property
    def has_audio(self):
        return bool(self.observed.any())

    @property
    def fully_exact(self):
        return bool(np.all(np.isin(self.labels, RATIOS)))


def allowed_states(label: int, mode: str):
    out = np.ones(K, dtype=bool)
    if mode == "unlabeled" or label not in RATIOS and label not in (8, 12):
        return out
    if mode == "exact":
        if label in RATIOS:
            out[:] = False
            out[RATIOS.index(label)] = True
        return out
    if mode == "family":
        if label == 1:
            out[:] = False; out[RATIOS.index(1)] = True
        elif label in (2, 4, 8):
            out[:] = False; out[RATIOS.index(2)] = True; out[RATIOS.index(4)] = True
        elif label in (3, 6, 12):
            out[:] = False; out[RATIOS.index(3)] = True; out[RATIOS.index(6)] = True
        return out
    raise ValueError(f"unknown constraint mode: {mode}")


def _build(seqs, source="mel50"):
    ev = classical.Evidence()
    ons, ys, ok = classical.interval_inputs(seqs, ev, source)
    result=[]; offset=0
    for rows in seqs:
        n=len(rows); y=ys[offset:offset+n].astype(np.float32,copy=True); observed=ok[offset:offset+n].astype(bool,copy=True)
        offset+=n
        # Unavailable features are not observations. Keep the interval/state transition,
        # but no density term will be added for these rows.
        y[~observed]=0.0
        labels=np.array([int(r['label']) for r in rows],dtype=np.int64)
        duration=np.array([float(r['perf_end'])-float(r['perf_start']) for r in rows],dtype=np.float32)
        duration=np.maximum(duration,1e-4)
        result.append(BeatSequence(
            work_id=rows[0]['work_id'], performance_id=rows[0]['performance_id'], split=rows[0]['split'],
            stem=rows[0]['feature_stem'],
            beat_index=np.array([int(r['beat_index']) for r in rows],dtype=np.int64),
            y=y, log_duration=np.log(duration), observed=observed, labels=labels,
            exact_allowed=np.stack([allowed_states(int(x),'exact') for x in labels]),
            family_allowed=np.stack([allowed_states(int(x),'family') for x in labels]),
            high_confidence=np.array([bool(r['high_confidence']) for r in rows]),
        ))
    return result


def load_sequences(source="mel50"):
    # Classical loader already segments on missing beat indices and applies the original work split.
    by_split={s:[] for s in ('train','val','test')}
    for seq in classical.load_sequences():
        by_split[seq[0]['split']].append(seq)
    out={s:_build(v,source=source) for s,v in by_split.items()}
    train_logdur=np.concatenate([q.log_duration[q.observed] for q in out['train'] if q.observed.any()])
    mu=float(train_logdur.mean()); sd=float(train_logdur.std()+1e-6)
    for split in out:
        for q in out[split]: q.log_duration=((q.log_duration-mu)/sd).astype(np.float32)
    return out, {'log_duration_mean':mu,'log_duration_std':sd,'feature_dim':int(out['train'][0].y.shape[1]),
                 'ratios':list(RATIOS),'train_sequences':len(out['train']),'val_sequences':len(out['val']),'test_sequences':len(out['test'])}


def choose_sequences(sequences, mode):
    """Training sequences and their constraint mode for a supervision regime."""
    audio = [q for q in sequences if q.has_audio]
    labeled = [q for q in audio if np.isin(q.labels, RATIOS).any()]
    if mode == "exact": return [q for q in audio if q.fully_exact], "exact"
    if mode == "supervised_only": return labeled, "exact"
    if mode == "mixed": return labeled + [q for q in audio if not np.isin(q.labels, RATIOS).any()], "exact"
    if mode == "partial_family": return audio, "family"
    if mode == "unsupervised": return audio, "unlabeled"
    raise ValueError(mode)


def validation_sequences(sequences, mode):
    """Validation sequences and constraint mode matching a training regime."""
    audio = [q for q in sequences if q.has_audio]
    if mode == "exact": return [q for q in audio if q.fully_exact], "exact"
    if mode in ("supervised_only", "mixed"): return [q for q in audio if np.isin(q.labels, RATIOS).any()], "exact"
    return audio, ("family" if mode == "partial_family" else "unlabeled")


def constraint_for(seq: BeatSequence, mode: Optional[str]):
    if mode is None or mode == 'unlabeled':
        return np.ones((seq.n,K),dtype=bool)
    if mode == 'exact': return seq.exact_allowed.copy()
    if mode == 'family': return seq.family_allowed.copy()
    raise ValueError(mode)


def sequence_summary(seqs):
    labels=np.concatenate([q.labels for q in seqs]) if seqs else np.zeros(0,dtype=int)
    observed=sum(int(q.observed.sum()) for q in seqs)
    missing=sum(int((~q.observed).sum()) for q in seqs)
    return {'sequences':len(seqs),'intervals':int(len(labels)),'observed_intervals':observed,'missing_intervals':missing,
            'fully_exact_sequences':sum(q.fully_exact for q in seqs),
            'sequences_with_any_exact_label':sum(np.isin(q.labels,RATIOS).any() for q in seqs),
            'label_counts':{str(r):int((labels==r).sum()) for r in sorted(set(labels.tolist()))}}
