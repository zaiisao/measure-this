"""Compare carried-meter page-OMR predictions with source MusicXML meter sequences."""
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "results/page_omr_pilot/carried_meter_batch"
METER_TOKEN = re.compile(r"\*M(\d+(?:\+\d+)*)/(\d+(?:\+\d+)*)")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def source_meter_sequence(path):
    path = Path(path)
    if path.suffix.lower() in (".mxl", ".zip"):
        with zipfile.ZipFile(path) as archive:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            xml_path = container.find(".//{*}rootfile").get("full-path")
            data = archive.read(xml_path)
    else:
        data = path.read_bytes()

    root = ET.fromstring(data)
    parts = root.findall("./{*}part")
    measures = [part.findall("{*}measure") for part in parts]
    current, sequence = "", []
    for index in range(max(map(len, measures), default=0)):
        meter = ""
        for part in measures:
            if index >= len(part):
                continue
            for time in part[index].findall("./{*}attributes/{*}time"):
                beats = "+".join(node.text or "" for node in time.findall("{*}beats"))
                unit = "+".join(node.text or "" for node in time.findall("{*}beat-type"))
                if beats and unit:
                    meter = f"{beats}/{unit}"
                    break
            if meter:
                break
        if meter:
            current = meter
        if current and (not sequence or sequence[-1] != current):
            sequence.append(current)
    return sequence


def page_meter_sequence(batch, row):
    text = (batch / row["prediction_file"]).read_text(errors="replace")
    meters = [f"{beats}/{unit}" for beats, unit in METER_TOKEN.findall(text)]
    return [meter for i, meter in enumerate(meters) if i == 0 or meter != meters[i - 1]]


def transitions(sequence):
    return list(zip(sequence, sequence[1:]))


def main():
    pages = read_csv(BATCH / "page_omr_results.csv")
    by_score = {}
    for row in pages:
        by_score.setdefault(row["score_id"], []).append(row)

    exact = opening_exact = completed = 0
    source_changes, predicted_changes = Counter(), Counter()
    mismatches = []
    for score_id, score_pages in sorted(by_score.items()):
        score_pages.sort(key=lambda row: int(row["page"]))
        source = source_meter_sequence(score_pages[0]["score_file"])
        predicted = []
        for row in score_pages:
            for meter in page_meter_sequence(BATCH, row):
                if not predicted or predicted[-1] != meter:
                    predicted.append(meter)

        done = (
            len(score_pages) == int(score_pages[0]["page_count"])
            and all(row["status"] == "ok" for row in score_pages)
        )

        completed += done
        if not done or not source or not predicted:
            continue

        source_changes.update(transitions(source))
        predicted_changes.update(transitions(predicted))

        opening_exact += source[0] == predicted[0]
        exact += source == predicted
        if source != predicted:
            mismatches.append((score_id, source, predicted))

    matched = sum((source_changes & predicted_changes).values())
    n_source, n_predicted = sum(source_changes.values()), sum(predicted_changes.values())

    print(f"Pages completed: {sum(row['status'] == 'ok' for row in pages)}/{len(pages)}")
    print(f"Score sequences completed: {completed}/{len(by_score)}")
    print(f"Opening meter exact: {opening_exact}/{completed}")
    print(f"Complete meter sequence exact: {exact}/{completed}")
    print(f"Meter transitions matched: {matched}/{n_source}; predicted={n_predicted}")
    print(f"Transition precision: {matched / n_predicted:.1%}; recall: {matched / n_source:.1%}")

    if mismatches:
        print("Non-exact score sequences:")
        for score_id, source, predicted in mismatches:
            print(f"  {score_id}: source {';'.join(source)}; OMR {';'.join(predicted)}")


if __name__ == "__main__":
    main()
