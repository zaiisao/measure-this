"""Onset evidence for one ASAP performance, in ASAP performance-MIDI time (= audio time - start).

Three sources, used as an ablation of the onset front end:
* audio100: librosa onset strength (log-mel, max-filtered "superflux"-style) at 100 fps on the
  MAESTRO audio excerpt, plus librosa peak-picked onset times.
* mel50: positive spectral flux of the 50 fps log-mel feature store used by Explore This, plus
  peak-picked onsets (same peak picker).
* perfmidi: performance-MIDI note-on times, chords merged within 30 ms. This is an *ideal onset
  detector* (played notes), used only as a diagnostic observation, never as a label.
"""
import io
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ASAP, EXPLORE_THIS_REPO, FEATURES, MAESTRO_ZIP  # noqa: E402,F401
from midi_io import read_midi  # noqa: E402

SR, HOP = 44100, 441  # 100 fps


def read_excerpt(member, start, end):
    with zipfile.ZipFile(MAESTRO_ZIP) as z, z.open(member) as fh:
        with wave.open(io.BytesIO(fh.read())) as w:
            sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
            x = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).mean(1) / 32768.0
    a = int(round(start * sr))
    b = int(round(end * sr)) if end else len(x)
    return x[a:b].astype(np.float32), sr


def peaks(env, fps):
    import librosa
    idx = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                 delta=0.07 * float(np.max(env) + 1e-9), wait=3)
    return idx / fps


def audio100(member, start, end):
    import librosa
    x, sr = read_excerpt(member, start, end)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP, n_fft=2048, n_mels=138, lag=1, max_size=3)
    env = env / (np.percentile(env, 99) + 1e-9)
    t = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="time", backtrack=False)
    return env.astype(np.float32), 100.0, np.asarray(t)


def mel50(stem):
    sys.path.insert(0, str(EXPLORE_THIS_REPO))
    from explore_this.dataset.mmnpz import MemmappedNpzFile
    spect = np.asarray(MemmappedNpzFile(str(FEATURES))[f"{stem}/track"], dtype=np.float32)
    env = np.concatenate([[0.0], np.maximum(np.diff(spect, axis=0), 0).mean(1)])
    env = env / (np.percentile(env, 99) + 1e-9)
    return env.astype(np.float32), 50.0, peaks(env, 50.0)


def perfmidi(perf_id):
    t = np.asarray(sorted(n.time for n in read_midi(ASAP / perf_id).notes))
    if not len(t):
        return t
    keep = [t[0]]
    for x in t[1:]:
        if x - keep[-1] > 0.03:
            keep.append(x)
    return np.asarray(keep)


def envelope_from_events(times, fps=100.0, duration=None, width=0.01):
    """Impulse-train envelope of discrete events (for running envelope models on MIDI onsets)."""
    n = int(np.ceil(((duration or (times.max() + 1)) * fps))) + 1
    env = np.zeros(n, np.float32)
    grid = np.arange(n) / fps
    for t in times:
        i = int(round(t * fps))
        lo, hi = max(0, i - 5), min(n, i + 6)
        env[lo:hi] += np.exp(-0.5 * ((grid[lo:hi] - t) / width) ** 2)
    return env
