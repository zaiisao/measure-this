"""Parse ASAP's timestamped performance time-signature annotations."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .common import MeterCandidate, MeterEvent, beat_files


REPO = Path(__file__).resolve().parents[2]
ASAP_ROOT = REPO / "third_party" / "asap-dataset"
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "asap" / "annotations" / "beats"
REPEAT_SUFFIX = re.compile(r"_(?:no_(?:first_|2_)?)?repeat(?=_|$)")


def _normalize_id(recording_id: str) -> str:
    return REPEAT_SUFFIX.sub("", recording_id).casefold()


def _asap_recording_id(row: dict[str, str]) -> str:
    folder = row["folder"].strip("/").replace("/", "_")
    performance_stem = Path(row["midi_performance"]).stem
    return f"{folder}_{performance_stem}"


def _performance_events(annotation: dict[str, Any]) -> list[MeterEvent]:
    changes = []
    for time_s, value in sorted(
        ((float(time_s), value) for time_s, value in annotation["perf_time_signatures"].items()),
        key=lambda item: item[0],
    ):
        signature, beats_per_bar = value
        if not changes or signature != changes[-1].signature:
            changes.append(
                MeterEvent(
                    time_s=time_s,
                    signature=str(signature),
                    beats_per_bar=int(beats_per_bar),
                )
            )
    return changes


def load_candidates(
    metadata_path: Path = ASAP_ROOT / "metadata.csv",
    annotations_path: Path = ASAP_ROOT / "asap_annotations.json",
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Join each Beat This ASAP file to ASAP's matching performance labels."""
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        metadata_rows = list(csv.DictReader(handle))
    with annotations_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)

    rows_by_normalized_id: dict[str, list[dict[str, str]]] = {}
    for row in metadata_rows:
        source_id = _asap_recording_id(row)
        rows_by_normalized_id.setdefault(_normalize_id(source_id), []).append(row)

    annotation_files = beat_files(beat_annotations_dir)
    candidates = []
    for recording_id, beat_path in sorted(annotation_files.items()):
        # Preserve the canonical on-disk spelling in pipeline output.
        recording_id = beat_path.stem
        matches = rows_by_normalized_id.get(_normalize_id(recording_id), [])
        if len(matches) != 1:
            reason = "no unique ASAP metadata match"
            candidates.append(
                MeterCandidate(
                    dataset="asap",
                    recording_id=recording_id,
                    beat_annotation_path=beat_path,
                    source_kind="asap_annotations.json:perf_time_signatures",
                    source_path=annotations_path,
                    source_status="unmatched" if not matches else "ambiguous",
                    reason=reason,
                )
            )
            continue

        row = matches[0]
        source_key = row["midi_performance"]
        annotation = annotations.get(source_key)
        if annotation is None or not annotation.get("perf_time_signatures"):
            candidates.append(
                MeterCandidate(
                    dataset="asap",
                    recording_id=recording_id,
                    beat_annotation_path=beat_path,
                    source_kind="asap_annotations.json:perf_time_signatures",
                    source_path=annotations_path,
                    source_recording_id=source_key,
                    source_status="missing_signature",
                    reason="ASAP performance has no perf_time_signatures annotation",
                    metadata={key: row.get(key, "") for key in ("composer", "title", "folder", "xml_score", "midi_performance")},
                )
            )
            continue

        events = _performance_events(annotation)
        candidates.append(
            MeterCandidate(
                dataset="asap",
                recording_id=recording_id,
                events=events,
                source_path=annotations_path,
                source_kind="asap_annotations.json:perf_time_signatures",
                source_recording_id=source_key,
                beat_annotation_path=beat_path,
                timestamped=True,
                constant_by_source=(len(events) <= 1),
                metadata={key: row.get(key, "") for key in ("composer", "title", "folder", "xml_score", "midi_performance")},
            )
        )
    return candidates
