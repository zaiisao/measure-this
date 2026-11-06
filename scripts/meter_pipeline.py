"""Transfer and audit time-signature labels across Beat This datasets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from meter_datasets import asap, filosax, groove_midi, harmonix
from meter_datasets.common import MeterCandidate, MeterEvent


REPO = Path(__file__).resolve().parents[1]
BEAT_THIS_ROOT = REPO / "third_party" / "beat_this_annotations"
HARMONIX_ROOT = REPO / "third_party" / "harmonixset"
GROOVE_ROOT = REPO / "third_party" / "groove_midi"
DEFAULT_OUT = REPO / "results" / "meter_pipeline"
MEASURE_THIS_ROOT = REPO / "measure_this_annotations"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def beat_annotation_files(dataset: str) -> list[Path]:
    directory = BEAT_THIS_ROOT / dataset / "annotations" / "beats"
    return sorted(directory.glob("*.beats"))


def candidates_without_source(
    dataset: str,
    reason: str,
    source_status: str = "source_unavailable",
) -> list[MeterCandidate]:
    return [
        MeterCandidate(
            dataset=dataset,
            recording_id=path.stem,
            beat_annotation_path=path,
            source_kind="unresolved_source_discovery",
            source_status=source_status,
            reason=reason,
        )
        for path in beat_annotation_files(dataset)
    ]


def load_direct_candidates(dataset: str) -> list[MeterCandidate]:
    if dataset == "asap":
        return asap.load_candidates()

    if dataset == "filosax":
        return filosax.load_candidates()

    if dataset == "harmonix":
        metadata_path = HARMONIX_ROOT / "dataset" / "metadata.csv"
        if not metadata_path.is_file():
            return candidates_without_source(
                dataset,
                f"Harmonix metadata file is missing: {metadata_path}",
            )
        return harmonix.load_candidates(metadata_path)

    if dataset == "groove_midi":
        metadata_path = GROOVE_ROOT / "info.csv"
        if not metadata_path.is_file():
            return candidates_without_source(
                dataset,
                "Groove MIDI is distributed as an archive, not an upstream git repository; "
                f"extract its info.csv and MIDI files under {GROOVE_ROOT}",
            )
        return groove_midi.load_candidates(metadata_path)

    return candidates_without_source(
        dataset,
        "No direct time-signature source is configured, and generic score discovery "
        "and score-to-performance alignment are not implemented yet.",
        source_status="unresolved",
    )


def read_beat_labels(path: Path) -> tuple[list[float], list[int], str]:
    times: list[float] = []
    positions: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return [], [], f"could not read Beat This annotation: {error}"

    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 2:
            continue
        try:
            time_s = float(fields[0])
            beat_position = int(fields[1])
        except ValueError:
            return [], [], f"invalid beat row at line {line_number}"
        times.append(time_s)
        positions.append(beat_position)

    return times, positions, ""


def possible_bar_lengths(event: MeterEvent) -> list[int]:
    if event.beats_per_bar is not None:
        return [event.beats_per_bar]

    try:
        numerator, denominator = (int(part) for part in event.signature.split("/", 1))
    except (TypeError, ValueError):
        return []

    possibilities = {numerator}
    if denominator == 8 and numerator > 3 and numerator % 3 == 0:
        possibilities.add(numerator // 3)
    return sorted(possibilities)


def event_for_time(events: list[MeterEvent], time_s: float) -> MeterEvent | None:
    active = None
    for event in events:
        if event.time_s is None or event.time_s <= time_s:
            active = event
        else:
            break
    return active


def sanity_check(candidate: MeterCandidate) -> dict[str, Any]:
    path = candidate.beat_annotation_path
    if path is None or not candidate.events:
        return {
            "status": "insufficient",
            "reason": "missing beat annotation or meter candidate",
            "downbeat_count": 0,
            "bar_length_distribution": {},
            "mismatches": [],
        }

    times, positions, error = read_beat_labels(path)
    if error:
        return {
            "status": "insufficient",
            "reason": error,
            "downbeat_count": 0,
            "bar_length_distribution": {},
            "mismatches": [],
        }

    downbeat_indices = [
        index for index, position in enumerate(positions) if position == 1
    ]
    if len(downbeat_indices) < 2:
        return {
            "status": "insufficient",
            "reason": "Beat This annotation has fewer than two labelled downbeats",
            "downbeat_count": len(downbeat_indices),
            "bar_length_distribution": {},
            "mismatches": [],
        }

    distribution: Counter[int] = Counter()
    mismatches = []
    for start_index, end_index in zip(downbeat_indices, downbeat_indices[1:]):
        observed_beats = end_index - start_index
        distribution[observed_beats] += 1
        start_time = times[start_index]
        event = event_for_time(candidate.events, start_time)
        if event is None:
            continue
        expected = possible_bar_lengths(event)
        if expected and observed_beats not in expected:
            mismatches.append(
                {
                    "start_downbeat_s": start_time,
                    "end_downbeat_s": times[end_index],
                    "signature": event.signature,
                    "observed_beat_count": observed_beats,
                    "expected_beat_counts": expected,
                }
            )

    return {
        "status": "compatible" if not mismatches else "incompatible",
        "reason": "" if not mismatches else "some downbeat intervals disagree with the candidate meter",
        "downbeat_count": len(downbeat_indices),
        "bars_checked": len(downbeat_indices) - 1,
        "bar_length_distribution": dict(sorted(distribution.items())),
        "mismatches": mismatches,
    }


def nearest_downbeat(
    time_s: float,
    times: list[float],
    positions: list[int],
) -> tuple[int | None, float | None, float | None]:
    downbeats = []
    for beat_time, position in zip(times, positions):
        if position == 1:
            downbeats.append((len(downbeats), beat_time))
    if not downbeats:
        return None, None, None
    downbeat_index, downbeat_time = min(
        downbeats,
        key=lambda item: abs(item[1] - time_s),
    )
    return downbeat_index, downbeat_time, abs(downbeat_time - time_s)


def copy_beat_this_annotations() -> None:
    """Copy the Beat This repository tree without its Git metadata."""
    MEASURE_THIS_ROOT.mkdir(parents=True, exist_ok=True)
    for source_path in BEAT_THIS_ROOT.iterdir():
        if source_path.name == ".git":
            continue

        destination_path = MEASURE_THIS_ROOT / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)


def signature_tags(candidate: MeterCandidate) -> tuple[dict[int, str], str]:
    """Return line numbers to annotate, placing changes on their downbeats."""
    source_path = candidate.beat_annotation_path
    if source_path is None or candidate.source_status != "matched" or not candidate.events:
        return {}, ""

    try:
        lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as error:
        return {}, f"could not read beat annotation: {error}"

    beat_rows: list[tuple[int, float, int]] = []
    for line_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            time_s = float(fields[0])
            beat_position = int(fields[1])
        except ValueError:
            return {}, f"invalid beat row at line {line_index + 1}"
        beat_rows.append((line_index, time_s, beat_position))

    if not beat_rows:
        return {}, "beat annotation contains no beat rows"

    tags = {beat_rows[0][0]: candidate.events[0].signature}
    downbeats = [row for row in beat_rows if row[2] == 1]
    for event in candidate.events[1:]:
        if event.time_s is None:
            return tags, "later meter event has no timestamp"
        if not downbeats:
            return tags, "beat annotation contains no labelled downbeat for meter change"

        line_index, downbeat_time, offset_s = min(
            downbeats,
            key=lambda row: abs(row[1] - event.time_s),
        )
        if abs(downbeat_time - event.time_s) > 0.1:
            return tags, (
                f"meter change at {event.time_s:.6f}s is {abs(downbeat_time - event.time_s):.6f}s "
                "from the nearest labelled downbeat"
            )
        if line_index in tags and tags[line_index] != event.signature:
            return tags, "multiple time signatures map to the same downbeat"
        tags[line_index] = event.signature

    return tags, ""


def write_signature_tags(path: Path, tags: dict[int, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    has_existing_third_column = any(len(line.split()) > 2 for line in lines)

    if has_existing_third_column:
        for line_index, line in enumerate(lines):
            newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            content = line[:-len(newline)] if newline else line
            fields = content.split()
            if len(fields) < 2:
                continue

            signature = tags.get(line_index, "")
            lines[line_index] = "\t".join(fields[:2] + [signature] + fields[2:]) + newline
        path.write_text("".join(lines), encoding="utf-8")
        return

    for line_index, signature in tags.items():
        line = lines[line_index]
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        content = line[:-len(newline)] if newline else line
        fields = content.split()
        if len(fields) >= 3:
            fields[2] = signature
            lines[line_index] = "\t".join(fields) + newline
        else:
            lines[line_index] = f"{content}\t{signature}{newline}"
    path.write_text("".join(lines), encoding="utf-8")


def annotate_copied_dataset(candidates: list[MeterCandidate]) -> tuple[int, int]:
    """Write time signatures into the copied Beat This beat files."""
    tagged_files = 0
    tagged_events = 0
    for candidate in candidates:
        if candidate.beat_annotation_path is None:
            continue

        tags, error = signature_tags(candidate)
        if error:
            print(f"Could not fully annotate {candidate.dataset}/{candidate.recording_id}: {error}")
        if not tags:
            continue

        relative_path = candidate.beat_annotation_path.relative_to(BEAT_THIS_ROOT)
        output_path = MEASURE_THIS_ROOT / relative_path
        write_signature_tags(output_path, tags)
        tagged_files += 1
        tagged_events += len(tags)

    return tagged_files, tagged_events


def process_candidate(
    candidate: MeterCandidate,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sanity = sanity_check(candidate)
    event_rows: list[dict[str, Any]] = []
    signatures = [event.signature for event in candidate.events]

    if candidate.source_status != "matched":
        status = (
            "source_unavailable"
            if candidate.source_status == "source_unavailable"
            else "unresolved"
        )
        reason = candidate.reason
    elif not candidate.events:
        status = "unresolved"
        reason = candidate.reason or "source did not yield a time-signature candidate"
    elif sanity["status"] == "insufficient":
        status = "review"
        reason = sanity["reason"]
    elif sanity["status"] == "incompatible":
        status = "review"
        reason = sanity["reason"]
    elif candidate.timestamped:
        status = "verified_changing" if len(candidate.events) > 1 else "verified_constant"
        reason = ""
        times, positions, _ = read_beat_labels(candidate.beat_annotation_path)
        for event_number, event in enumerate(candidate.events):
            if event_number == 0 or event.time_s is None:
                continue
            downbeat_index, downbeat_time, offset_s = nearest_downbeat(
                event.time_s,
                times,
                positions,
            )
            if offset_s is None or offset_s > 0.1:
                status = "review"
                reason = "meter-change timestamp does not match a Beat This downbeat"
                event_rows = []
                break
            event_rows.append(
                {
                    "dataset": candidate.dataset,
                    "recording_id": candidate.recording_id,
                    "previous_signature": candidate.events[event_number - 1].signature,
                    "signature": event.signature,
                    "change_time_s": event.time_s,
                    "downbeat_index": downbeat_index,
                    "downbeat_time_s": downbeat_time,
                    "alignment_offset_s": offset_s,
                    "source_path": str(candidate.source_path or ""),
                }
            )
    elif candidate.constant_by_source is True:
        status = "verified_constant"
        reason = ""
    else:
        status = "review"
        reason = (
            "source provides a recording-level signature but does not establish "
            "that the written meter stays constant throughout"
        )

    if not signatures:
        time_signature = ""
    elif len(signatures) == 1:
        time_signature = signatures[0]
    else:
        time_signature = "changing"

    row = {
        "dataset": candidate.dataset,
        "recording_id": candidate.recording_id,
        "status": status,
        "time_signature": time_signature,
        "time_signatures": signatures,
        "review_reason": reason,
        "source_status": candidate.source_status,
        "source_kind": candidate.source_kind,
        "source_recording_id": candidate.source_recording_id or "",
        "source_path": str(candidate.source_path or ""),
        "beat_annotation_path": str(candidate.beat_annotation_path or ""),
        "timestamped_source": candidate.timestamped,
        "sanity_status": sanity["status"],
        "sanity_reason": sanity["reason"],
        "downbeat_count": sanity["downbeat_count"],
        "bars_checked": sanity.get("bars_checked", 0),
        "bar_length_distribution": sanity["bar_length_distribution"],
        "sanity_mismatches": sanity["mismatches"],
        "source_metadata": candidate.metadata,
    }
    return row, event_rows


def signature_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        signature
        for row in rows
        for signature in set(row["time_signatures"])
    )
    return dict(sorted(counts.items()))


def run(out_dir: Path) -> dict[str, Any]:
    copy_beat_this_annotations()
    dataset_names = sorted(
        path.name
        for path in BEAT_THIS_ROOT.iterdir()
        if path.is_dir() and (path / "annotations" / "beats").is_dir()
    )

    recordings: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    tagged_files = 0
    tagged_events = 0
    for dataset in dataset_names:
        candidates = load_direct_candidates(dataset)
        dataset_tagged_files, dataset_tagged_events = annotate_copied_dataset(candidates)
        tagged_files += dataset_tagged_files
        tagged_events += dataset_tagged_events
        for candidate in candidates:
            row, candidate_events = process_candidate(candidate)
            recordings.append(row)
            events.extend(candidate_events)
            provenance.append(candidate.as_dict())

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "recording_annotations.csv", recordings)
    write_csv(out_dir / "time_signature_events.csv", events)
    write_csv(out_dir / "provenance.csv", provenance)
    write_csv(
        out_dir / "review_queue.csv",
        [row for row in recordings if row["status"] == "review"],
    )
    write_csv(
        out_dir / "unresolved_recordings.csv",
        [row for row in recordings if row["status"] in {"unresolved", "source_unavailable"}],
    )

    status_counts = Counter(row["status"] for row in recordings)
    dataset_summary = {}
    for dataset in dataset_names:
        rows = [row for row in recordings if row["dataset"] == dataset]
        verified_rows = [
            row
            for row in rows
            if row["status"] in {"verified_constant", "verified_changing"}
        ]
        dataset_summary[dataset] = {
            "recordings": len(rows),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "candidate_time_signature_coverage": signature_coverage(rows),
            "verified_time_signature_coverage": signature_coverage(verified_rows),
        }
    verified_recordings = [
        row
        for row in recordings
        if row["status"] in {"verified_constant", "verified_changing"}
    ]
    summary = {
        "recordings": len(recordings),
        "statuses": dict(status_counts),
        "time_signature_events": len(events),
        "measure_this_annotations": str(MEASURE_THIS_ROOT),
        "tagged_recordings": tagged_files,
        "time_signature_labels_written": tagged_events,
        "candidate_time_signature_coverage": signature_coverage(recordings),
        "verified_time_signature_coverage": signature_coverage(verified_recordings),
        "datasets": dataset_summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
