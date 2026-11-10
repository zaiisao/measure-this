"""Parse HJDB recordings using the dataset's documented 4/4 meter."""

from __future__ import annotations

from pathlib import Path

from .common import MeterCandidate, MeterEvent, beat_files


REPO = Path(__file__).resolve().parents[2]
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "hjdb" / "annotations" / "beats"
MIREX_URL = "https://music-ir.org/mirex/wiki/2021:Audio_Downbeat_Estimation"
PAPER_URL = "https://ismir2012.ismir.net/event/papers/169_ISMIR_2012.pdf"


def load_candidates(
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Label HJDB excerpts as 4/4, as specified by the MIREX dataset page."""
    candidates = []
    for _, beat_path in sorted(beat_files(beat_annotations_dir).items()):
        candidates.append(
            MeterCandidate(
                dataset="hjdb",
                recording_id=beat_path.stem,
                events=[MeterEvent(time_s=None, signature="4/4", beats_per_bar=4)],
                source_path=None,
                source_kind="dataset_documentation:time_signature",
                source_recording_id=beat_path.stem,
                beat_annotation_path=beat_path,
                timestamped=False,
                constant_by_source=True,
                metadata={
                    "source_url": MIREX_URL,
                    "source_description": "MIREX describes all HJDB excerpts as 4/4.",
                    "paper_url": PAPER_URL,
                    "paper_qualification": (
                        "The original paper says HJDB is almost exclusively in 4/4 "
                        "and uses a four-beat downbeat-period assumption."
                    ),
                    "beat_position_audit": (
                        "All 235 Beat This annotation files contain metric positions "
                        "1 through 4, cycling 1-2-3-4; this supports four beat positions "
                        "per bar but does not independently identify the denominator."
                    ),
                },
            )
        )
    return candidates
