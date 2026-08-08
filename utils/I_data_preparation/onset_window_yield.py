#!/usr/bin/env python3

# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Yield of the trigger-free windowing, per recording.
===================================================

For every windowed ``.h5`` file it prints how many windows the onset detector
actually produced against how many utterances the recording contains, in
absolute numbers and as a percentage.

The reference count is taken from the prepared recording of the same session
(``raw_and_processed/<subject>/<condition>/sess_N_batch_M.h5``) as the number of
contiguous runs of a non-zero trigger label, that is the number of cue boxes the
protocol presented. The detector never reads that trigger, so the ratio is the
recall of the detector on that recording.

Only base windows are counted: augmented copies, if the dataset carries any, are
excluded through ``augmentation_shift_ms``. Rest windows (label 0) are excluded
as well, so the figure compares speech against speech.

Usage
-----
    python utils/I_data_preparation/onset_window_yield.py \
        --data_dir data_sentences/data_sentences_5_subjects_6_sessions \
        --win_dir wins_and_features_2_4

    # restrict to some subjects / conditions, or to one window folder
    python utils/I_data_preparation/onset_window_yield.py \
        --data_dir data_sentences/data_sentences_5_subjects_6_sessions \
        --win_dir wins_and_features_2_4 \
        --subjects S01 S02 --conditions silent --window WIN_2400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def count_cue_boxes(prepared_h5: Path) -> Optional[int]:
    """Number of contiguous runs of a non-zero trigger label in a recording."""
    try:
        df = pd.read_hdf(prepared_h5)
    except Exception as exc:  # pragma: no cover - depends on the stored file
        print(f"    [warn] cannot read {prepared_h5}: {exc}", file=sys.stderr)
        return None

    if "Label_int" not in df.columns:
        print(f"    [warn] no Label_int in {prepared_h5}", file=sys.stderr)
        return None

    lab = df["Label_int"].to_numpy()
    if lab.size == 0:
        return 0
    starts = np.flatnonzero(np.diff(lab) != 0) + 1
    heads = np.concatenate(([0], starts))
    return int(np.count_nonzero(lab[heads] != 0))


def count_windows(win_h5: Path) -> Optional[int]:
    """Number of base speech windows stored in a windowed file."""
    try:
        df = pd.read_hdf(win_h5, "wins_feats")
    except Exception as exc:  # pragma: no cover - depends on the stored file
        print(f"    [warn] cannot read {win_h5}: {exc}", file=sys.stderr)
        return None

    if "augmentation_shift_ms" in df.columns:
        df = df[df["augmentation_shift_ms"] == 0]
    if "Label_int" in df.columns:
        df = df[df["Label_int"] != 0]
    return int(len(df))


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def iter_window_files(
    data_dir: Path,
    win_dir: str,
    processed_dir: str,
    subjects: Optional[List[str]],
    conditions: Optional[List[str]],
    window: Optional[str],
):
    """Yield ``(subject, condition, window_folder, win_h5, prepared_h5)``."""
    root = data_dir / win_dir
    if not root.is_dir():
        raise SystemExit(f"no such windowed dataset: {root}")

    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if subjects and subj_dir.name not in subjects:
            continue
        for cond_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            if conditions and cond_dir.name not in conditions:
                continue
            for win_folder in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
                if window and win_folder.name != window:
                    continue
                for win_h5 in sorted(win_folder.glob("*.h5")):
                    prepared = (
                        data_dir
                        / processed_dir
                        / subj_dir.name
                        / cond_dir.name
                        / win_h5.name
                    )
                    yield subj_dir.name, cond_dir.name, win_folder.name, win_h5, prepared


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[1])
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data_sentences/data_sentences_5_subjects_6_sessions"),
        help="Dataset root holding the prepared recordings and the window folders.",
    )
    ap.add_argument(
        "--win_dir",
        default="wins_and_features_2_4",
        help="Windowed dataset folder inside --data_dir (the trigger-free one).",
    )
    ap.add_argument(
        "--processed_dir",
        default="raw_and_processed",
        help="Folder holding the prepared recordings, used for the reference count.",
    )
    ap.add_argument("--subjects", nargs="+", default=None, help="e.g. S01 S02")
    ap.add_argument("--conditions", nargs="+", default=None, help="e.g. silent vocalized")
    ap.add_argument("--window", default=None, help="Restrict to one window folder, e.g. WIN_2400")
    ap.add_argument("--csv", type=Path, default=None, help="Optional path for a CSV of the per-file rows.")
    args = ap.parse_args(argv)

    rows: List[Dict[str, object]] = []
    per_group: Dict[Tuple[str, str], List[int]] = {}

    print()
    print(f"Trigger-free windowing yield: {args.data_dir / args.win_dir}")
    print(f"reference = cue boxes in {args.data_dir / args.processed_dir}")
    print()
    print(f"{'file':<52}{'kept':>8}{'total':>8}{'yield':>10}")
    print("-" * 78)

    current_group: Optional[Tuple[str, str]] = None
    for subject, condition, win_folder, win_h5, prepared in iter_window_files(
        args.data_dir, args.win_dir, args.processed_dir, args.subjects, args.conditions, args.window
    ):
        kept = count_windows(win_h5)
        total = count_cue_boxes(prepared) if prepared.is_file() else None
        if kept is None:
            continue

        group = (subject, condition)
        if group != current_group:
            if current_group is not None:
                print()
            print(f"[{subject} / {condition} / {win_folder}]")
            current_group = group

        name = f"  {win_h5.name}"
        if total:
            pct = 100.0 * kept / total
            print(f"{name:<52}{kept:>8}{total:>8}{pct:>9.1f}%")
            per_group.setdefault(group, [0, 0])
            per_group[group][0] += kept
            per_group[group][1] += total
        else:
            print(f"{name:<52}{kept:>8}{'n/a':>8}{'n/a':>10}")

        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "window": win_folder,
                "file": win_h5.name,
                "windows_kept": kept,
                "cue_boxes": total if total is not None else "",
                "yield_percent": round(100.0 * kept / total, 2) if total else "",
            }
        )

    if not rows:
        print("no windowed files found")
        return 1

    print()
    print("=" * 78)
    print(f"{'subject / condition':<52}{'kept':>8}{'total':>8}{'yield':>10}")
    print("-" * 78)
    tot_kept = tot_all = 0
    for (subject, condition), (kept, total) in sorted(per_group.items()):
        pct = 100.0 * kept / total if total else float("nan")
        print(f"  {subject} / {condition:<41}{kept:>8}{total:>8}{pct:>9.1f}%")
        tot_kept += kept
        tot_all += total
    print("-" * 78)
    if tot_all:
        print(f"  {'ALL':<43}{tot_kept:>8}{tot_all:>8}{100.0 * tot_kept / tot_all:>9.1f}%")
        print(f"\n  {tot_all - tot_kept} of {tot_all} utterances were not detected "
              f"({100.0 * (tot_all - tot_kept) / tot_all:.1f}%)")
    print()

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"[SAVED] {args.csv}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
