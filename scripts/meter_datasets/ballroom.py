"""Parse Ballroom meter from bar annotations and published dance-class labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import MeterCandidate, MeterEvent, beat_files


REPO = Path(__file__).resolve().parents[2]
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "ballroom" / "annotations" / "beats"
PAPER_URL = "https://archives.ismir.net/ismir2016/paper/000132.pdf"
ANNOTATIONS_URL = "https://github.com/CPJKU/BallroomAnnotations"


def _read_positions(path: Path) -> list[int]:
    """Read beat-in-bar IDs from one Ballroom ``.beats`` file."""
    positions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            positions.append(int(fields[1]))
        except ValueError as error:
            raise ValueError(f"invalid beat ID on line {line_number}") from error
    return positions


def _infer_signature(path: Path) -> tuple[str | None, int | None, str]:
    """Use annotated bar lengths to select the paper's 3/4 or 4/4 class."""
    try:
        positions = _read_positions(path)
    except (OSError, UnicodeError, ValueError) as error:
        return None, None, f"could not read beat-in-bar annotations: {error}"

    downbeats = [index for index, position in enumerate(positions) if position == 1]
    if len(downbeats) < 2:
        return None, None, "fewer than two annotated downbeats; bar length cannot be checked"

    lengths = [end - start for start, end in zip(downbeats, downbeats[1:])]
    distribution = Counter(lengths)
    if len(distribution) != 1:
        return None, None, f"inconsistent annotated bar lengths: {dict(sorted(distribution.items()))}"

    beats_per_bar = next(iter(distribution))
    if any(
        positions[start:end] != list(range(1, beats_per_bar + 1))
        for start, end in zip(downbeats, downbeats[1:])
    ):
        return None, beats_per_bar, "beat IDs do not form complete consecutive bars"
    if beats_per_bar not in (3, 4):
        return None, beats_per_bar, f"observed {beats_per_bar} beats per bar, outside the paper's Ballroom labels"

    # The published Ballroom table assigns the 3-beat Waltz classes 3/4 and
    # the six 4-beat classes 4/4. The Beat This IDs omit dance-style folders,
    # so the bar cycle selects between those published class labels.
    return f"{beats_per_bar}/4", beats_per_bar, ""


def load_candidates(
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Label Ballroom excerpts using annotated bar length and the 2016 paper."""
    candidates = []
    for _, beat_path in sorted(beat_files(beat_annotations_dir).items()):
        signature, beats_per_bar, reason = _infer_signature(beat_path)
        candidates.append(
            MeterCandidate(
                dataset="ballroom",
                recording_id=beat_path.stem,
                events=(
                    [MeterEvent(time_s=None, signature=signature, beats_per_bar=beats_per_bar)]
                    if signature is not None
                    else []
                ),
                source_path=None,
                source_kind="published_dance_class_meter+annotated_bar_length",
                source_recording_id=beat_path.stem,
                beat_annotation_path=beat_path,
                timestamped=False,
                constant_by_source=True if signature is not None else None,
                source_status="matched" if signature is not None else "unresolved",
                reason=reason,
                metadata={
                    "paper_url": PAPER_URL,
                    "paper_citation": (
                        "Holzapfel and Grill, Bayesian Meter Tracking on Learned Signal "
                        "Representations, ISMIR 2016, Table 1. The table labels Cha Cha, "
                        "Jive, Quickstep, Rumba, Samba, and Tango as 4/4; Waltz and "
                        "Viennese Waltz as 3/4."
                    ),
                    "beat_annotations_url": ANNOTATIONS_URL,
                    "assignment_method": (
                        "Beat This Ballroom filenames omit dance-style folders. The "
                        "number of beats between annotated downbeats selects the 3/4 "
                        "or 4/4 class listed in the paper."
                    ),
                    "observed_beats_per_bar": beats_per_bar,
                },
            )
        )
    return candidates
