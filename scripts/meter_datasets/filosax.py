"""Parse FiloSax beat files using the collection's fixed 4/4 meter."""

from __future__ import annotations

from pathlib import Path

from .common import MeterCandidate, MeterEvent, beat_files


REPO = Path(__file__).resolve().parents[2]
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "filosax" / "annotations" / "beats"
SOURCE_URL = "https://www.eecs.qmul.ac.uk/~simond/pub/2021/FosterDixon-Filosax-ISMIR2021.pdf"


def load_candidates(
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Label FiloSax recordings as 4/4 and retain beat files for sanity checks."""
    candidates = []
    for recording_id, beat_path in sorted(beat_files(beat_annotations_dir).items()):
        candidates.append(
            MeterCandidate(
                dataset="filosax",
                recording_id=beat_path.stem,
                events=[MeterEvent(time_s=None, signature="4/4", beats_per_bar=4)],
                source_path=None,
                source_kind="collection_protocol",
                source_recording_id=beat_path.stem,
                beat_annotation_path=beat_path,
                timestamped=False,
                constant_by_source=True,
                metadata={
                    "collection_meter_rule": "all pieces were selected to be entirely in 4/4",
                    "source_url": SOURCE_URL,
                },
            )
        )
    return candidates
