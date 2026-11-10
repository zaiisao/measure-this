"""Parse JAAH's recording-level metre annotations."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .common import MeterCandidate, MeterEvent, beat_files, normalize_signature


REPO = Path(__file__).resolve().parents[2]
JAAH_ANNOTATIONS_ROOT = REPO / "third_party" / "jaah" / "annotations"
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations" / "jaah" / "annotations" / "beats"
BEAT_ID_PREFIX = re.compile(r"^jaah_\d+-", re.IGNORECASE)


def _normalize_identity(value: str) -> str:
    """Normalize names for matching source artist/title to Beat This IDs."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return "".join(character.casefold() for character in ascii_text if character.isalnum())


def _read_annotation(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            annotation = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return annotation if isinstance(annotation, dict) else None


def load_candidates(
    annotations_dir: Path = JAAH_ANNOTATIONS_ROOT,
    beat_annotations_dir: Path = BEAT_THIS_ROOT,
) -> list[MeterCandidate]:
    """Join JAAH's ``metre`` fields to Beat This recordings by artist/title."""
    annotations_by_identity: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        annotation = _read_annotation(annotation_path)
        if annotation is None:
            continue
        artist = annotation.get("artist")
        title = annotation.get("title")
        if not isinstance(artist, str) or not isinstance(title, str):
            continue
        identity = _normalize_identity(f"{artist}{title}")
        if identity:
            annotations_by_identity.setdefault(identity, []).append((annotation_path, annotation))

    candidates = []
    for _, beat_path in sorted(beat_files(beat_annotations_dir).items()):
        recording_id = beat_path.stem
        source_identity = _normalize_identity(BEAT_ID_PREFIX.sub("", recording_id))
        matches = annotations_by_identity.get(source_identity, [])
        if len(matches) != 1:
            candidates.append(
                MeterCandidate(
                    dataset="jaah",
                    recording_id=recording_id,
                    beat_annotation_path=beat_path,
                    source_path=annotations_dir,
                    source_kind="annotation:metre",
                    source_status="unmatched" if not matches else "ambiguous",
                    reason="no unique JAAH annotation matched by artist and title",
                )
            )
            continue

        annotation_path, annotation = matches[0]
        raw_signature = annotation.get("metre", "")
        try:
            signature = normalize_signature(str(raw_signature))
        except ValueError as error:
            candidates.append(
                MeterCandidate(
                    dataset="jaah",
                    recording_id=recording_id,
                    beat_annotation_path=beat_path,
                    source_path=annotation_path,
                    source_kind="annotation:metre",
                    source_recording_id=str(annotation.get("mbid") or "") or None,
                    source_status="missing_signature",
                    reason=str(error),
                    metadata={
                        key: annotation.get(key, "")
                        for key in ("artist", "title", "mbid", "duration")
                    },
                )
            )
            continue

        candidates.append(
            MeterCandidate(
                dataset="jaah",
                recording_id=recording_id,
                events=[MeterEvent(time_s=None, signature=signature)],
                source_path=annotation_path,
                source_kind="annotation:metre",
                source_recording_id=str(annotation.get("mbid") or "") or None,
                beat_annotation_path=beat_path,
                timestamped=False,
                # JAAH supplies one recording-level metre, not a meter timeline.
                # Keep it as candidate evidence for the pipeline's review step.
                constant_by_source=None,
                metadata={
                    key: annotation.get(key, "")
                    for key in ("artist", "title", "mbid", "duration")
                },
            )
        )
    return candidates
