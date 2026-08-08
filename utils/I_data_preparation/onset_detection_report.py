#!/usr/bin/env python3

# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Report for the trigger-free onset detector?
===================================================================

Scores ``utils/I_data_preparation/onset_detection.py`` against the trigger boxes
of the processed recordings. The detector never sees the trigger; it is used here
only as ground truth, so the numbers say how much would be lost by cutting the
windows without it.

Read the columns as follows:

* ``det`` — how many cue boxes received at least one onset, as a percentage and
  as the plain count. Whatever is missing is the set of utterances that would
  simply not appear in the onset-aligned training set.
* ``FA`` — how many detected events overlap no cue box at all. Swallows and
  movements, mostly. An utterance split in two is not counted here, so this is
  spurious events only, not detection mistakes.
* ``dur p95`` — how long the detected utterances last, from onset to offset.
  "p95" means 95% of them are shorter than this, so a window of that length
  fits all but the longest 5%. It is the number to pick ``window_size_s`` from:
  choose less and you cut the tail off more than 5% of the utterances.

Usage
-----
    python utils/I_data_preparation/onset_detection_report.py \\
        --data_dir data_sentences

    # one subject, all its sessions, and a CSV of the per-recording table
    python utils/I_data_preparation/onset_detection_report.py \\
        --data_dir data_sentences \\
        --subjects S01 --out onset_report.csv

    # try a different operating point before committing to an extraction
    python utils/I_data_preparation/onset_detection_report.py \\
        --data_dir data_sentences \\
        --t_high 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.I_data_preparation.experimental_config import RAW_AND_FILTERED_DIRNAME
from utils.I_data_preparation.onset_detection import (
    OnsetConfig,
    detect_events_in_dataframe,
    label_boxes,
    score_against_trigger,
)


def discover(data_dir: Path, processed: str, subjects, conditions) -> List[Path]:
    root = data_dir / processed
    if not root.is_dir():
        raise FileNotFoundError(f"No processed recordings under '{root}'.")
    files: List[Path] = []
    for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if subjects and sub_dir.name not in subjects:
            continue
        for cond_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
            if conditions and cond_dir.name not in conditions:
                continue
            files.extend(sorted(cond_dir.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 recordings matched the selection under '{root}'.")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--processed", default=RAW_AND_FILTERED_DIRNAME)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--t_high", type=float, default=None, help="Override OnsetConfig.t_high.")
    ap.add_argument("--t_low_ratio", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Write the per-recording table to CSV.")
    args = ap.parse_args()

    overrides = {}
    if args.t_high is not None:
        overrides["t_high"] = args.t_high
    if args.t_low_ratio is not None:
        overrides["t_low_ratio"] = args.t_low_ratio
    cfg = OnsetConfig(**overrides)

    files = discover(args.data_dir, args.processed, args.subjects, args.conditions)
    print(f"Scoring {len(files)} recordings | t_high={cfg.t_high} t_low={cfg.t_low:.2f} "
          f"topk={cfg.topk} min_dur={cfg.min_duration_s}s\n")

    header = f"{'recording':<34} {'det':>18} {'FA':>6} {'dur p95':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for path in files:
        df = pd.read_hdf(path, key="emg")
        df = pd.DataFrame(df).reset_index(drop=True)
        events = detect_events_in_dataframe(df, cfg)
        boxes = label_boxes(df)
        stats = score_against_trigger(events, boxes, len(df), fs=cfg.fs)

        subject, condition = path.parents[1].name, path.parent.name
        name = f"{subject}/{condition}/{path.stem}"
        rows.append({"subject": subject, "condition": condition, "recording": path.stem, **stats})
        det = f"{stats['detection_rate']:.0%} ({stats['n_detected']}/{stats['n_boxes']})"
        print(f"{name:<34} {det:>18} {stats['false_alarms']:>6} {stats['duration_p95_s']:>9.2f}")

    table = pd.DataFrame(rows)
    print("-" * len(header))
    n_det, n_box = int(table["n_detected"].sum()), int(table["n_boxes"].sum())
    total = f"{n_det / n_box:.0%} ({n_det}/{n_box})" if n_box else "n/a"
    print(
        f"{'TOTAL':<34} {total:>18} {int(table['false_alarms'].sum()):>6} "
        f"{table['duration_p95_s'].mean():>9.2f}"
    )
    print(f"\n{n_box - n_det} of {n_box} utterances were not detected "
          f"({(n_box - n_det) / n_box:.1%})." if n_box else "")
    worst = table.loc[table["detection_rate"].idxmin()]
    print(
        f"\nWorst recording: {worst['subject']}/{worst['condition']}/{worst['recording']} "
        f"at {worst['detection_rate']:.0%} detection."
    )
    print(
        f"Suggested window_size_s to cover the 95th percentile of utterances: "
        f"{np.ceil(table['duration_p95_s'].mean() * 10) / 10:.1f} s "
        f"(per-recording p95 ranges {table['duration_p95_s'].min():.2f}-"
        f"{table['duration_p95_s'].max():.2f} s)."
    )

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nPer-recording table written to {args.out}")


if __name__ == "__main__":
    main()
