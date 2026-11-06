"""Parse Harmonix's recording-level time-signature metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from .common import MeterCandidate, MeterEvent, beat_files, normalize_signature


REPO = Path(__file__).resolve().parents[2]
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "harmonix" / "annotations" / "beats"


def _load_rows(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel_tab if metadata_path.suffix.casefold() == ".tsv" else csv.excel
        return list(csv.DictReader(handle, dialect=dialect))


def _find_value(row: dict[str, str], name: str) -> str:
    return next((value for key, value in row.items() if key.strip().casefold() == name.casefold()), "")


def _dataset_number(recording_id: str) -> str | None:
    prefix = recording_id.split("_", maxsplit=1)[0]
    return prefix if prefix.isdigit() else None


def load_candidates(
    metadata_path: Path,
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Join Harmonix metadata's ``File``/``Time Signature`` fields to beat files."""
    rows_by_id: dict[str, list[dict[str, str]]] = {}
    rows_by_number: dict[str, list[dict[str, str]]] = {}
    for row in _load_rows(metadata_path):
        source_id = Path(_find_value(row, "File")).stem
        if source_id:
            rows_by_id.setdefault(source_id.casefold(), []).append(row)
            number = _dataset_number(source_id)
            if number is not None:
                rows_by_number.setdefault(number, []).append(row)

    candidates = []
    for recording_id, beat_path in sorted(beat_files(beat_annotations_dir).items()):
        matches = rows_by_id.get(recording_id, [])
        if not matches:
            number = _dataset_number(recording_id)
            if number is not None:
                matches = rows_by_number.get(number, [])
        if len(matches) != 1:
            candidates.append(
                MeterCandidate(
                    dataset="harmonix",
                    recording_id=beat_path.stem,
                    beat_annotation_path=beat_path,
                    source_path=metadata_path,
                    source_kind="metadata:Time Signature",
                    source_status="unmatched" if not matches else "ambiguous",
                    reason="no unique Harmonix metadata row for recording ID",
                )
            )
            continue

        row = matches[0]
        raw_signature = _find_value(row, "Time Signature")
        try:
            signature = normalize_signature(raw_signature)
        except ValueError as error:
            candidates.append(
                MeterCandidate(
                    dataset="harmonix",
                    recording_id=beat_path.stem,
                    beat_annotation_path=beat_path,
                    source_path=metadata_path,
                    source_kind="metadata:Time Signature",
                    source_recording_id=_find_value(row, "File"),
                    source_status="missing_signature",
                    reason=str(error),
                )
            )
            continue

        candidates.append(
            MeterCandidate(
                dataset="harmonix",
                recording_id=beat_path.stem,
                events=[MeterEvent(time_s=None, signature=signature)],
                source_path=metadata_path,
                source_kind="metadata:Time Signature",
                source_recording_id=_find_value(row, "File"),
                beat_annotation_path=beat_path,
                timestamped=False,
                # This track-level value is candidate evidence, not a timeline
                # proving that no changes occur within the song.
                constant_by_source=None,
                metadata={key: value for key, value in row.items() if key.strip().casefold() != "time signature"},
            )
        )
    return candidates
