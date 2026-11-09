# page-omr workflow

The page-omr work has two reusable commands and an image helper:

1. `infer.py` runs pages in score order and carries the previous page's meter into continuation-page headers. It also defines `prepare_page_image`, which composites transparent pixels onto white before conversion to RGB. Printed meter changes after the first barline remain model-decoded. Output goes under `results/page_omr_pilot/carried_meter_batch/`.
2. `evaluate.py` compares saved `**kern` predictions directly with their source MusicXML meter sequences. It prints completion, opening-meter, sequence, and transition results; it writes no files.

To evaluate the carried-meter predictions:

```sh
python3 scripts/page_omr/evaluate.py
```

The alignment experiment has a separate script for its detailed audit tables:

```sh
python3 results/metrical_structure_alignment/evaluate_omr_sequences.py
```

The independent-page inference launcher was removed; its existing predictions remain preserved. Run inference with `python3 scripts/page_omr/infer.py`. The evaluator reads the saved page manifest and prediction files under `carried_meter_batch/`, plus the original MusicXML scores.
