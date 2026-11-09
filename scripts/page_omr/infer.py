"""page-omr inference that carries the time signature from page to page.

page-omr transcribes one page at a time, and its **kern header must state the meter in effect at
the start of the page. On page 2 onward that meter comes from earlier pages the model never sees,
so it guesses. Here each score's pages run in order: page 1 decodes normally, and on every later
page the header meter tokens (those before the first bar line) are forced to the last meter of the
previous page. Meters after a bar line, which come from printed signatures, decode normally.
"""
import argparse
import csv
import multiprocessing as mp
import re
import sys
from pathlib import Path

from PIL import Image

MODEL = "tobiashornbogen/page-omr-a1-kern"
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "third_party" / "page-omr" / "src"))


def prepare_page_image(image: Image.Image) -> Image.Image:
    """Composite transparent page pixels onto white before RGB conversion."""
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


class CarriedMeter:
    """Logits processor: before the first bar line, replace any meter token with the carried one."""

    def __init__(self, tokenizer, meter_ids, carried):
        self.tokenizer = tokenizer
        self.meter_ids = meter_ids
        self.forced = meter_ids.get(carried)
        self.text = ""
        self.length = 1

    def __call__(self, generated, logits, step, max_length):
        new = generated[0, self.length:].tolist()
        if new:
            self.text += self.tokenizer.decode(new)
            self.length = generated.size(1)
        in_header = not re.search(r"(^|\n)=", self.text)
        if self.forced is not None and in_header and int(logits[0].argmax()) in self.meter_ids.values():
            logits[0, :] = float("-inf")
            logits[0, self.forced] = 0.0
        return logits


def worker(gpu, scores, queue, batch, predictions):
    import torch
    from omr.inference.postprocess import postprocess_output
    from omr.inference.repair import repair_kern_prediction
    from omr.utils.hub import load_from_hub

    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    model, tokenizer, processor, _ = load_from_hub(MODEL, device=device)
    model = model.half().eval()

    meter_ids = {}
    for token, token_id in tokenizer.get_vocab().items():
        match = re.fullmatch(r"\*M(\d+/\d+)", tokenizer.decode([token_id]).strip())
        if match:
            meter_ids[match.group(1)] = token_id

    for pages in scores:
        carried = None
        for row in pages:
            with Image.open(batch / row["image"]) as source:
                pixels = processor(prepare_page_image(source), return_tensors="pt")["pixel_values"]

            forcing = CarriedMeter(tokenizer, meter_ids, carried) if carried else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                ids = model.generate(
                    pixels.to(device).half(),
                    max_length=3072,
                    num_beams=1,
                    repetition_penalty=1.0,
                    logits_processor=forcing,
                )
            decoded = tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
            postprocessed = postprocess_output(decoded)
            text = repair_kern_prediction(postprocessed)
            path = predictions / f"{Path(row['image']).stem}.krn"
            path.write_text(text)
            queue.put({
                "score_file": row["score_file"],
                "score_id": row["score_id"],
                "page": row["page"],
                "page_count": row["page_count"],
                "prediction_file": str(path.relative_to(predictions.parent)),
                "status": "ok",
            })
            meters = re.findall(r"\*M(\d+/\d+)", text)
            carried = meters[-1] if meters else carried

    queue.put(None)


def load_score_shards(batch, worker_count):
    with (batch / "page_omr_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))

    by_score = {}
    for row in rows:
        by_score.setdefault(row["score_id"], []).append(row)
    scores = [sorted(pages, key=lambda r: int(r["page"])) for pages in by_score.values()]
    scores.sort(key=len, reverse=True)
    shards = [[] for _ in range(worker_count)]
    shard_sizes = [0] * worker_count
    for pages in scores:
        shard = min(range(worker_count), key=shard_sizes.__getitem__)
        shards[shard].append(pages)
        shard_sizes[shard] += len(pages)
    return rows, shards


def run_workers(shards, batch, predictions, total_pages):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=worker, args=(gpu, shard, queue, batch, predictions))
        for gpu, shard in enumerate(shards)
    ]

    for process in processes:
        process.start()

    results, finished = [], 0
    while finished < len(processes):
        message = queue.get()
        if message is None:
            finished += 1
            continue
        results.append(message)
        if len(results) % 50 == 0:
            print(f"completed {len(results)}/{total_pages}", flush=True)

    for process in processes:
        process.join()

    results.sort(key=lambda r: (r["score_id"], int(r["page"])))
    return results


def write_summary(path, results):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Run page-OMR inference with carried meters.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of GPU workers to launch (default: 4)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    batch = PROJECT / "results/page_omr_pilot/collected_score_batch"
    output = PROJECT / "results/page_omr_pilot/carried_meter_batch"
    predictions = output / "predictions"
    summary = output / "page_omr_results.csv"

    predictions.mkdir(parents=True, exist_ok=True)

    rows, shards = load_score_shards(batch, args.workers)
    results = run_workers(shards, batch, predictions, len(rows))
    write_summary(summary, results)
    print(f"DONE pages={len(results)} summary={summary}", flush=True)


if __name__ == "__main__":
    main()
