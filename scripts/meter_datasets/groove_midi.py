"""Parse Groove MIDI's native ``info.csv`` time-signature field."""

from __future__ import annotations

import csv
from pathlib import Path

from .common import MeterCandidate, MeterEvent, beat_files, normalize_signature


REPO = Path(__file__).resolve().parents[2]
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "groove_midi" / "annotations" / "beats"


def load_candidates(
    metadata_path: Path,
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Join Groove ``info.csv`` rows to Beat This files by MIDI filename stem."""
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        metadata_rows = list(csv.DictReader(handle))

    annotations_by_id: dict[str, list[dict[str, str]]] = {}
    for row in metadata_rows:
        midi_filename = row.get("midi_filename", "")
        source_id = Path(midi_filename).with_suffix("").as_posix().replace("/", "_")
        annotations_by_id.setdefault(source_id.casefold(), []).append(row)

    annotation_files = beat_files(beat_annotations_dir)
    candidates = []
    for recording_id, beat_path in sorted(annotation_files.items()):
        matches = annotations_by_id.get(recording_id, [])
        if len(matches) != 1:
            candidates.append(
                MeterCandidate(
                    dataset="groove_midi",
                    recording_id=beat_path.stem,
                    beat_annotation_path=beat_path,
                    source_path=metadata_path,
                    source_kind="info.csv:time_signature",
                    source_status="unmatched" if not matches else "ambiguous",
                    reason="no unique Groove metadata row for MIDI filename",
                )
            )
            continue

        row = matches[0]
        try:
            signature = normalize_signature(row["time_signature"])
        except (KeyError, ValueError) as error:
            candidates.append(
                MeterCandidate(
                    dataset="groove_midi",
                    recording_id=beat_path.stem,
                    beat_annotation_path=beat_path,
                    source_path=metadata_path,
                    source_kind="info.csv:time_signature",
                    source_recording_id=row.get("midi_filename", ""),
                    source_status="missing_signature",
                    reason=str(error),
                    metadata={key: value for key, value in row.items() if key != "time_signature"},
                )
            )
            continue

        candidates.append(
            MeterCandidate(
                dataset="groove_midi",
                recording_id=beat_path.stem,
                events=[MeterEvent(time_s=None, signature=signature)],
                source_path=metadata_path,
                source_kind="info.csv:time_signature",
                source_recording_id=row.get("midi_filename", ""),
                beat_annotation_path=beat_path,
                timestamped=False,
                # The per-performance value is candidate evidence, not a meter
                # timeline proving that no changes occur within the file.
                constant_by_source=None,
                metadata={key: value for key, value in row.items() if key != "time_signature"},
            )
        )
    return candidates
