# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
General Utilities for Offline Experiments
"""

import sys
from pathlib import Path
import re
import json
from datetime import datetime
import torch
import numpy as np
import random
from typing import List, Union
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.II_feature_extraction.FeatExtractorManager import FeatureRegistry
from models.seeds import TORCH_MANUAL_SEED, RANDOM_SEED, RGN_SEED
from utils.I_data_preparation.read_bio_file import parse_bio_filename
from utils.I_data_preparation.experimental_config import (
    RAW_DIRNAME,
    RAW_AND_FILTERED_DIRNAME,
    WINS_AND_FEATURES_DIRNAME,
    WINDOW_DIR_PREFIX,
    SILENT_DIRNAME,
    VOCALIZED_DIRNAME,
)

# Matches the session id in processed/window filenames (sess_<id>_batch_<id>.h5).
SESSION_RE = re.compile(r"sess_(\d+)")


#################################### Utils for Data Preparation ######################################


def feature_names_to_consider(
    consider_time_feats: bool = True,
    consider_freq_feats: bool = True,
    consider_wavelet_feats: bool = True,
) -> List[str]:
    """
    Returns the base feature names to consider depending on flags.
    """
    features = []

    if consider_time_feats:
        features += FeatureRegistry.TIME_DOMAIN

    if consider_freq_feats:
        features += FeatureRegistry.FREQUENCY_DOMAIN

    if consider_wavelet_feats:
        features += FeatureRegistry.WAVELET_DOMAIN

    return features


def feature_columns_to_consider(feature_names: List[str], df: pd.DataFrame) -> List[str]:
    """
    Returns only the DataFrame columns corresponding to selected base feature names.

    feature_names: list[str]
    """

    if not feature_names:
        raise ValueError("No feature names provided!")

    # <feature>_<win>_Ch_<idx>_filt
    pattern = r"^(" + "|".join(map(re.escape, feature_names)) + r")_\d+_Ch_\d+_filt$"

    selected_cols = [c for c in df.columns if re.search(pattern, c)]
    return selected_cols


def reorder_ml_features_by_channel(cols: List[str], channel_order: List[int]) -> List[str]:
    """
    Reorder ML feature columns based on channel_order.
    Feature names must contain pattern: _Ch_<idx>_filt
    """

    order_position = {ch: i for i, ch in enumerate(channel_order)}

    parsed = []
    for col in cols:
        match = re.search(r"_Ch_(\d+)", col)
        if match:
            ch_idx = int(match.group(1))
            ch_rank = order_position.get(ch_idx, 10**9)
        else:
            # if no channel info, push to end
            ch_rank = 10**9

        parsed.append((ch_rank, col))

    # stable sort: python sort is stable → preserves feature grouping inside channel
    parsed_sorted = sorted(parsed, key=lambda x: x[0])

    return [col for _, col in parsed_sorted]


#################################### Utils to override configs ######################################


def deep_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with dict u (u wins)."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


#################### Utils to Keep Track of Runs ##############################


def mark_running(run_dir: Path, meta: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUNNING").write_text(datetime.now().isoformat())
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def mark_done(run_dir: Path):
    running = run_dir / "RUNNING"
    if running.exists():
        running.unlink()
    (run_dir / "DONE").write_text(datetime.now().isoformat())


def mark_failed(run_dir: Path, err: str):
    (run_dir / "FAILED").write_text(err)
    # keep RUNNING as evidence if you want; or remove it:
    running = run_dir / "RUNNING"
    if running.exists():
        running.unlink()


def should_skip(run_dir: Path, *, rerun_failed=False, rerun_running=False) -> bool:
    if (run_dir / "DONE").exists():
        return True
    if (run_dir / "RUNNING").exists() and not rerun_running:
        return True
    if (run_dir / "FAILED").exists() and not rerun_failed:
        return True
    return False


#################### Utils for data loading ################################


## TO-DO: check which functions use this and replace with load_function in utils/general_utils.py
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def build_wins_feats_dirs(
    main_data_directory: Path,
    base_config: dict,
    sub_id: Union[str, List[str]],
    condition: str,
    window_size_ms: int,
) -> List[Path]:
    """Build the windows/features directories for the requested subject(s)/condition.

    The windows folder name is taken from base_config["paths"]["win_and_feats"]
    when present, otherwise from the canonical WINS_AND_FEATURES_DIRNAME constant,
    keeping a single source of truth for the folder layout.
    """
    win_root = base_config.get("paths", {}).get("win_and_feats", WINS_AND_FEATURES_DIRNAME)
    subjects = [sub_id] if isinstance(sub_id, str) else list(sub_id)
    conditions = (
        [SILENT_DIRNAME, VOCALIZED_DIRNAME] if condition == "voc_and_silent" else [condition]
    )
    win_subdir = f"{WINDOW_DIR_PREFIX}{window_size_ms}"

    dirs: List[Path] = []
    for subject in subjects:
        for cond in conditions:
            dirs.append(Path(main_data_directory) / win_root / str(subject) / str(cond) / win_subdir)
    return dirs


def check_data_directories(
    main_data_directory: Path,
    all_subjects_models: bool,
    sub_id: Union[str, List[str]],
    condition: str,
    window_size_ms: int,
    base_config: dict,
) -> List[Path]:
    """
    Returns data directories containing data for training, depending on the desired training config.

    Returns
    -------
    List[Path]
        List of valid data directories.

    Raises
    ------
    FileNotFoundError
        If any expected directory does not exist.
    """
    data_dirs = build_wins_feats_dirs(
        main_data_directory, base_config, sub_id, condition, window_size_ms
    )

    # ---- existence check ----
    missing = [p for p in data_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The following data directories do not exist:\n" + "\n".join(str(p) for p in missing)
        )

    return data_dirs


def base_window_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Returns only the rows corresponding to the base window (i.e., no augmentation)."""
    if "augmentation_direction" not in df.columns:
        return df.copy()
    return df[df["augmentation_direction"] == "base"].copy()


def training_rows_with_augmentation(
    augmented_size_df: pd.DataFrame,
    train_base_df: pd.DataFrame,
    mode: str = "augmented_size",
    seed: int = 0,
) -> pd.DataFrame:
    """Returns the rows for training.

    - mode='augmented_size': keep all augmented rows derived from the base training split.
    - mode='original_size': sample the same number of rows as the original training split
      from the augmented pool, preserving label balance as much as possible.
    """
    if "augmentation_source_id" not in augmented_size_df.columns:
        return train_base_df.copy()

    train_source_ids = train_base_df["augmentation_source_id"].drop_duplicates()
    candidate_df = augmented_size_df[augmented_size_df["augmentation_source_id"].isin(train_source_ids)].copy()

    if mode == "original_size":
        target_size = int(len(train_base_df))
        if target_size < len(candidate_df):
            print(
                f"[DEBUG] original_size training mode:"
                f"\nAugmented candidate windows={len(candidate_df)}. "
                f"\nSampled windows={target_size}."
                f"\nOriginal base windows={len(train_base_df)}."
            )
            stratify = candidate_df["Label_int"] if "Label_int" in candidate_df.columns else None
            sample_ratio = target_size / len(candidate_df)
            sampled_train, _ = train_test_split(
                candidate_df,
                train_size=sample_ratio,
                random_state=seed,
                stratify=stratify,
                shuffle=True,
            )
            return sampled_train

    return candidate_df


def reset_all_seeds():
    """
    Resets all random seeds for PyTorch, NumPy, and Python's random module
    to ensure deterministic behavior for experiments.
    """
    torch.manual_seed(TORCH_MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_MANUAL_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    np.random.seed(RGN_SEED)
    random.seed(RANDOM_SEED)
    

def discover_sessions(data_dir: Path, subject: str, condition: str) -> List[int]:
    """Discover available session IDs for the given subject and condition.

    The primary source is the filtered recordings (``data_raw_and_filt``), which
    are part of the public dataset and are named ``sess_<id>_batch_<id>.h5``.
    Falls back to the raw ``.bio`` recordings (``raw``) for self-collected data.
    """
    data_dir = Path(data_dir)
    sessions = set()

    filt_dir = data_dir / RAW_AND_FILTERED_DIRNAME / subject / condition
    if filt_dir.exists():
        for h5_path in filt_dir.glob("*.h5"):
            match = SESSION_RE.search(h5_path.name)
            if match:
                sessions.add(int(match.group(1)))

    if not sessions:
        raw_dir = data_dir / RAW_DIRNAME / subject / condition
        if raw_dir.exists():
            for bio_path in sorted(raw_dir.glob("*.bio")):
                parsed = parse_bio_filename(bio_path)
                if parsed is None:
                    continue
                session_id, _, _ = parsed
                sessions.add(int(session_id))

    return sorted(sessions)
