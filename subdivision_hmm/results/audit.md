# Dataset audit

* ASAP performances: 1067; aligned with exact MAESTRO audio: 501; works: 169.
* Split by work: train 116, val 35, test 18 works (test = works of Explore This baseline_val ASAP).
* Labelled beat intervals: 258863; unknown (off-lattice attacks or <2 beats): 8728 (3.4%); of off-lattice attacks in unknown measures, 266/16788 lie within 0.1 beat of a beat (grace/arpeggio-like).
* Performances with any xml tuplet: 328.
* Beat-level vs measure-level label agreement (beat label = lattice of that beat's own attacks): 85.6% of labelled intervals.

| split | label | intervals | high-confidence | works with label |
|---|---|---:|---:|---:|
| train | unknown | 6307 | 0 | 77 |
| train | 1 | 23898 | 4594 | 70 |
| train | 2 | 33765 | 10459 | 60 |
| train | 3 | 21056 | 9325 | 53 |
| train | 4 | 36268 | 13655 | 79 |
| train | 6 | 23212 | 3580 | 58 |
| train | 8 | 8931 | 714 | 63 |
| train | 12 | 6500 | 84 | 62 |
| train | 16 | 1774 | 16 | 28 |
| train | 24 | 5749 | 60 | 51 |
| val | unknown | 2161 | 0 | 24 |
| val | 1 | 4961 | 1683 | 20 |
| val | 2 | 13591 | 8975 | 22 |
| val | 3 | 6663 | 1153 | 13 |
| val | 4 | 21379 | 13438 | 26 |
| val | 6 | 5339 | 568 | 20 |
| val | 8 | 1612 | 584 | 18 |
| val | 12 | 2202 | 0 | 22 |
| val | 16 | 109 | 0 | 7 |
| val | 24 | 1213 | 0 | 16 |
| test | unknown | 260 | 0 | 6 |
| test | 1 | 3540 | 2728 | 7 |
| test | 2 | 5729 | 5165 | 6 |
| test | 3 | 5768 | 5132 | 6 |
| test | 4 | 13250 | 11169 | 14 |
| test | 6 | 1052 | 288 | 8 |
| test | 8 | 854 | 278 | 6 |
| test | 12 | 1032 | 0 | 7 |
| test | 16 | 400 | 4 | 4 |
| test | 24 | 288 | 0 | 3 |
