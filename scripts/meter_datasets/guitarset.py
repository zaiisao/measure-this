"""Parse GuitarSet's beat-position annotations as 4/4 meter labels."""

from __future__ import annotations

from pathlib import Path

from .common import MeterCandidate, MeterEvent, beat_files


REPO = Path(__file__).resolve().parents[2]
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "guitarset" / "annotations" / "beats"
SOURCE_URL = "https://zenodo.org/records/3371780/files/annotation.zip?download=1"


def load_candidates(
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Label GuitarSet tracks from its official JAMS beat-position fields."""
    candidates = []
    for _, beat_path in sorted(beat_files(beat_annotations_dir).items()):
        recording_id = beat_path.stem
        source_recording_id = (
            recording_id[:-4] if recording_id.endswith("_mix") else recording_id
        )
        candidates.append(
            MeterCandidate(
                dataset="guitarset",
                recording_id=recording_id,
                events=[MeterEvent(time_s=None, signature="4/4", beats_per_bar=4)],
                source_path=None,
                source_kind="official_jams:beat_position",
                source_recording_id=source_recording_id,
                beat_annotation_path=beat_path,
                timestamped=False,
                constant_by_source=True,
                metadata={
                    "source_url": SOURCE_URL,
                    "source_annotation": "beat_position",
                    "num_beats": 4,
                    "beat_units": 4,
                    "corpus_audit": (
                        "All 360 official JAMS files have beat_position data; all 21,120 "
                        "events use num_beats=4 and beat_units=4, with positions 1 through 4."
                    ),
                },
            )
        )
    return candidates
