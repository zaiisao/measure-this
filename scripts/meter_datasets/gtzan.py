"""Parse meter labels from the GTZAN-Rhythm annotation archive."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .common import MeterCandidate, MeterEvent, beat_files, normalize_signature


REPO = Path(__file__).resolve().parents[2]
GTZAN_RHYTHM_ROOT = REPO / "third_party" / "GTZAN-Rhythm_v2_ismir2015_lbd"
STATS_PATH = GTZAN_RHYTHM_ROOT / "stats.csv"
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "gtzan" / "annotations" / "beats"
GTZAN_FILENAME = re.compile(r"^(?P<genre>[^.]+)\.(?P<number>\d+)\.wav$", re.IGNORECASE)


def _beat_this_id(filename: str) -> str | None:
    match = GTZAN_FILENAME.match(Path(filename).name)
    if match is None:
        return None
    genre = match.group("genre").casefold()
    number = int(match.group("number"))
    return f"gtzan_{genre}_{number:05d}"


def _load_rows(stats_path: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_id: dict[str, list[dict[str, str]]] = {}
    with stats_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            recording_id = _beat_this_id(row.get("filename", ""))
            if recording_id is not None:
                rows_by_id.setdefault(recording_id, []).append(row)
    return rows_by_id


def load_candidates(
    stats_path: Path = STATS_PATH,
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Join GTZAN-Rhythm meter values to Beat This recordings by GTZAN ID."""
    rows_by_id = _load_rows(stats_path)
    candidates = []

    for recording_id, beat_path in sorted(beat_files(beat_annotations_dir).items()):
        matches = rows_by_id.get(recording_id, [])
        if len(matches) != 1:
            candidates.append(
                MeterCandidate(
                    dataset="gtzan",
                    recording_id=beat_path.stem,
                    beat_annotation_path=beat_path,
                    source_path=stats_path,
                    source_kind="GTZAN-Rhythm:stats.csv:meter",
                    source_status="unmatched" if not matches else "ambiguous",
                    reason="no unique GTZAN-Rhythm stats row for recording ID",
                )
            )
            continue

        row = matches[0]
        raw_signature = row.get("meter", "").strip()
        metadata = {
            key: row.get(key, "")
            for key in (
                "artist",
                "title",
                "tempo mean",
                "tempo std",
                "swing ?",
                "ternary ?",
                "beat by measure",
            )
        }
        try:
            signature = normalize_signature(raw_signature)
        except ValueError:
            candidates.append(
                MeterCandidate(
                    dataset="gtzan",
                    recording_id=beat_path.stem,
                    beat_annotation_path=beat_path,
                    source_path=stats_path,
                    source_kind="GTZAN-Rhythm:stats.csv:meter",
                    source_recording_id=row.get("filename") or None,
                    source_status="missing_signature",
                    reason=(
                        "GTZAN-Rhythm leaves meter blank when the annotated beats per "
                        "measure are not constant or cannot be determined"
                    ),
                    metadata=metadata,
                )
            )
            continue

        candidates.append(
            MeterCandidate(
                dataset="gtzan",
                recording_id=beat_path.stem,
                events=[MeterEvent(time_s=None, signature=signature)],
                source_path=stats_path,
                source_kind="GTZAN-Rhythm:stats.csv:meter",
                source_recording_id=row.get("filename") or None,
                beat_annotation_path=beat_path,
                timestamped=False,
                constant_by_source=True,
                metadata=metadata,
            )
        )
    return candidates
