"""Shared records and small helpers for dataset-specific meter parsers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeterEvent:
    """One written/performance meter label and its optional source timestamp."""

    time_s: float | None
    signature: str
    beats_per_bar: int | None = None


@dataclass
class MeterCandidate:
    """A dataset's meter evidence for one Beat This recording."""

    dataset: str
    recording_id: str
    events: list[MeterEvent] = field(default_factory=list)
    source_path: Path | None = None
    source_kind: str = ""
    source_recording_id: str | None = None
    beat_annotation_path: Path | None = None
    timestamped: bool = False
    constant_by_source: bool | None = None
    source_status: str = "matched"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for pipeline output."""
        result = asdict(self)
        for key in ("source_path", "beat_annotation_path"):
            value = result[key]
            result[key] = str(value) if value is not None else None
        return result


def beat_files(directory: Path) -> dict[str, Path]:
    """Index Beat This annotation files by case-folded recording stem."""
    return {path.stem.casefold(): path for path in sorted(directory.glob("*.beats"))}


def normalize_signature(value: str) -> str:
    """Normalize common dataset spellings such as ``4-4`` and ``4|4``."""
    value = value.strip().replace("-", "/").replace("|", "/")
    if not value or value.casefold() in {"nan", "none", "unknown", "n/a"}:
        raise ValueError("missing time signature")
    if "/" not in value:
        raise ValueError(f"not a written time signature: {value!r}")
    numerator, denominator = value.split("/", maxsplit=1)
    return f"{numerator.strip()}/{denominator.strip()}"


def match_beat_file(
    recording_id: str,
    indexed_beat_files: dict[str, Path],
) -> tuple[Path | None, str]:
    """Return a unique Beat This path, allowing a final extension difference."""
    path = indexed_beat_files.get(recording_id.casefold())
    if path is None:
        return None, "no Beat This annotation with the source recording ID"
    return path, ""
