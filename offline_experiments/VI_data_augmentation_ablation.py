#!/usr/bin/env python3

"""
Data-augmentation ablation for offline experiments.

This script sweeps combinations of `data_augmentation` parameters, regenerates
windows/features for each combination, and runs either `global` or `inter_session`.
Saves results to cv_summary.csv for later plotting.
"""

from __future__ import annotations

import argparse
import tempfile
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.I_global_models import Global_Model_Trainer
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import reset_all_seeds
from utils.I_data_preparation.read_bio_file import parse_bio_filename
from utils.general_utils import SubjectConfig, load_all_h5files_from_folder, window_ms_from_cfg
from utils.II_feature_extraction.win_feature_extraction_main import Global_Windower_and_Feature_Extractor


class Data_Augmentation_Ablation_Trainer:
    def __init__(
        self,
        base_config: dict,
        model_config: dict,
        window_config: dict,
        data_dir: Path,
        artifacts_dir: Path,
        subjects: List[str],
        conditions: List[str],
        experiments: List[str],
        windows_s: List[float],
        stride_ms: List[int],
        num_strides: List[int],
    ) -> None:
        
        self.base_cfg = deepcopy(base_config)
        self.model_cfg = deepcopy(model_config)
        self.window_cfg_template = deepcopy(window_config)
        
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.working_data_root = self.artifacts_dir / ".ablation_working_data"
        
        self.subjects = subjects
        self.conditions = conditions
        self.experiments = experiments if experiments else ["global", "inter_session"]
        
        self.windows_s = windows_s if windows_s else [1.4]
        
        self.stride_ms_values: List[int] = [int(v) for v in (stride_ms if stride_ms else [10])]
        self.num_strides_values: List[int] = [int(v) for v in (num_strides if num_strides else [2, 5, 10])]
        self.base_stride: int = self.stride_ms_values[0] if self.stride_ms_values else 10
        
        self.ablation_results: List[Dict[str, Any]] = []
        
    @staticmethod
    def _normalize_window_size_s(window_value: float) -> float:
        """Normalizes the window size to seconds if it's given in milliseconds."""
        window_value = float(window_value)
        return window_value / 1000.0

    def _discover_sessions(self, subject: str, condition: str) -> List[int]:
        """Discovers available session IDs for the given subject and condition by parsing BIO filenames in the raw data directory."""
        raw_dir = self.data_dir / "raw" / subject / condition
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

        sessions = set()
        for bio_path in sorted(raw_dir.glob("*.bio")):
            parsed = parse_bio_filename(bio_path)
            if parsed is None:
                continue
            session_id, _, _ = parsed
            sessions.add(int(session_id))

        sessions_sorted = sorted(sessions)
        if not sessions_sorted:
            raise ValueError(f"No valid BIO session files found in: {raw_dir}")
        return sessions_sorted

    def _ensure_combo_data_root(self, combo_root: Path) -> Path:
        """Ensures the combo_root directory exists and contains symlinks to the shared raw and raw_and_processed data folders."""
        combo_root.mkdir(parents=True, exist_ok=True)
        for folder_name in ["raw", "raw_and_processed"]:
            target = combo_root / folder_name
            source = (self.data_dir / folder_name).resolve()
            if target.exists() or target.is_symlink():
                target.unlink()
            if not source.exists():
                raise FileNotFoundError(f"Missing shared source folder: {source}")
            target.symlink_to(source, target_is_directory=True)
        return combo_root

    def _run_windowing(
        self,
        run_window_cfg: dict,
        combo_data_root: Path,
        subject: str,
        condition: str,
        window_s: float,
        data_augmentation: dict
    ) -> bool:
        """
        Runs the windowing and feature extraction for the given configuration. 
        Returns True if successful, False if skipped due to no windows.
        """
        normalized_window_s = self._normalize_window_size_s(window_s)
        cfg = deepcopy(run_window_cfg)
        cfg.setdefault("data", {})
        cfg.setdefault("window", {})
        cfg["data"]["data_directory"] = str(combo_data_root)
        cfg["data"]["subject_id"] = str(subject)
        cfg["condition"] = str(condition)
        cfg["window"]["window_size_s"] = normalized_window_s
        cfg["data_augmentation"] = deepcopy(data_augmentation)

        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = Path(td) / "create_windows_tmp.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
            subject_cfg = SubjectConfig(tmp_cfg)
            extractor = Global_Windower_and_Feature_Extractor(subject_cfg)
            try:
                extractor.main()
                return True
            except KeyError as e:
                if "session_id" in str(e) or "batch_id" in str(e):
                    print("\n[WARNING] Extractor returned 0 windows. Skipping extraction.")
                    return False
                raise

    def _run_global_model(
        self,
        cfg_run: dict,
        combo_data_root: Path,
        artifacts_n: Path,
        subject: str,
        condition: str,
        model_name_id: str,
        selected_sessions: List[int],
        data_augmentation: dict
    ) -> Path:
        """Runs the global experiment for the given configuration and returns the path to the model directory."""
        cfg_run["data"]["data_directory"] = str(combo_data_root)
        cfg_run["data"]["models_main_directory"] = str(artifacts_n)
        cfg_run["data"]["subject_id"] = str(subject)
        cfg_run["condition"] = str(condition)
        cfg_run["model_name_id"] = str(model_name_id)
        cfg_run["data_augmentation"] = deepcopy(data_augmentation)
        cfg_run.setdefault("experiment", {})["augmentation_ablation"] = {
            "num_sessions": int(len(selected_sessions)),
            "session_ids": [int(s) for s in selected_sessions],
            "data_augmentation": deepcopy(data_augmentation),
        }

        trainer = Global_Model_Trainer(base_config=cfg_run, model_config=deepcopy(self.model_cfg))
        df_full = load_all_h5files_from_folder(trainer.data_dire_proc[0], key="wins_feats", print_statistics=False)
        if df_full.empty: 
            raise ValueError("The loaded DataFrame is completely empty.")

        trainer.df = df_full[df_full["session_id"].isin(selected_sessions)].copy().reset_index(drop=True)
        if trainer.df.empty:
            raise ValueError("No rows left after filtering sessions.")

        trainer._save_run_cfg()
        trainer.run_cv()
        pd.DataFrame(trainer.cv_summaries).to_csv(trainer.model_dire / "cv_summary.csv", index=False)
        return trainer.model_dire

    def _run_inter_session_model(
        self,
        cfg_run: dict,
        combo_data_root: Path,
        artifacts_n: Path,
        subject: str,
        condition: str,
        model_name_id: str,
        selected_sessions: List[int],
        data_augmentation: dict
    ) -> Path:
        """
        Runs the inter-session experiment for the given configuration and returns the path to the model directory. 
        Raises ValueError if the experiment is deemed untrainable (e.g., not enough sessions).
        """
        cfg_run["data"]["data_directory"] = str(combo_data_root)
        cfg_run["data"]["models_main_directory"] = str(artifacts_n)
        cfg_run["data"]["subject_id"] = str(subject)
        cfg_run["condition"] = str(condition)
        cfg_run["model_name_id"] = str(model_name_id)
        cfg_run["data_augmentation"] = deepcopy(data_augmentation)
        cfg_run.setdefault("experiment", {})["augmentation_ablation"] = {
            "num_sessions": int(len(selected_sessions)),
            "session_ids": [int(s) for s in selected_sessions],
            "data_augmentation": deepcopy(data_augmentation),
        }

        trainer = Inter_Session_Model_Trainer(
            base_config=cfg_run, model_config=deepcopy(self.model_cfg), experiment_subdir="inter_session"
        )
        
        df_full = load_all_h5files_from_folder(trainer.data_dire_proc[0], key="wins_feats", print_statistics=False)
        if df_full.empty:
            raise ValueError("The loaded DataFrame is empty.")
        if "session_id" not in df_full.columns:
            raise ValueError("Loaded windows/features do not contain the required 'session_id' column.")
        trainer.df = df_full[df_full["session_id"].isin(selected_sessions)].copy().reset_index(drop=True)

        if trainer.df.empty:
            raise ValueError("No rows left after filtering sessions.")

        if len(np.sort(trainer.df["session_id"].unique())) < 2:
            raise ValueError("inter_session requires at least 2 sessions.")

        trainer._save_run_cfg()
        val_size = float(cfg_run.get("cv", {}).get("val_size", 0.3))
        seed = int(cfg_run.get("experiment", {}).get("seed", 0))
        trainer.run_inter_session_cv(val_size=val_size, seed=seed)
        pd.DataFrame(trainer.cv_summaries).to_csv(trainer.model_dire / "cv_summary.csv", index=False)
        return trainer.model_dire

    def _evaluate_single_experiment(
        self,
        current_exp: str,
        cfg_run: dict,
        combo_data_root: Path,
        artifacts_n: Path,
        subject: str,
        condition: str,
        model_name_id: str,
        selected_sessions: List[int],
        data_augmentation: dict,
        run_label: str,
        window_s: float,
        num_strides: int,
        n_sessions: int
    ) -> Optional[dict]:
        """Runs the specified experiment and returns a summary dict for results collection, or None if skipped/untrainable."""
        try:
            reset_all_seeds()

            if current_exp == "global":
                out_dir = self._run_global_model(
                    cfg_run, combo_data_root, artifacts_n, subject, condition,
                    model_name_id, selected_sessions, data_augmentation
                )
            elif current_exp == "inter_session":
                if len(selected_sessions) < 2:
                    print(f"[SKIP][inter_session] needs >=2 sessions")
                    return None
                out_dir = self._run_inter_session_model(
                    cfg_run, combo_data_root, artifacts_n, subject, condition,
                    model_name_id, selected_sessions, data_augmentation
                )
            else:
                return None
                
            print(f"[DONE][{current_exp}] {out_dir}")
            return {"experiment": current_exp, "run_label": run_label, "n_sessions": n_sessions}

        except ValueError as e:
            print(f"[SKIP] Un-trainable for {run_label} ({n_sessions} sessions) - {current_exp}: {e}")
            return None

    def main(self) -> None:
        augmentation_combos: Sequence[Tuple[Optional[int], Optional[int]]] = [(None, None)] + list(product(self.stride_ms_values, self.num_strides_values))
        print(f"\n[ABLATION] Running over {len(augmentation_combos)} combos (incl. baseline).")

        # 1. Loop on the combinations of data augmentation parameters (including baseline with no augmentation)
        for stride_ms, num_strides in augmentation_combos:

            # Use 'or' instead of 'and' so the type checker narrows the else block to strictly 'int'
            if stride_ms is None or num_strides is None:
                data_augmentation = {"mode": "disabled"}
                safe_num_strides = 0
                run_label = "baseline"
            else:
                s_val = int(stride_ms)
                n_val = int(num_strides)
                data_augmentation = {
                    "mode": "sliding_window", "stride_ms": s_val, "num_strides": n_val
                }
                safe_num_strides = n_val
                run_label = f"stride{s_val}_n{n_val}"

            print(f"\n{'='*90}")
            print(f"[STARTING VARIANT] -> {run_label.upper()}")
            print(f"{'='*90}")

            combo_data_root = self._ensure_combo_data_root(self.working_data_root / run_label)

            for subject in self.subjects:
                for condition in self.conditions:
                    sessions = self._discover_sessions(subject, condition)
                    max_sessions_available = len(sessions)
                    if max_sessions_available == 0:
                        print(f"  [SKIP] No sessions found for {subject} {condition}")
                        continue

                    for window_s in self.windows_s:
                        run_window_cfg = deepcopy(self.window_cfg_template)
                        run_window_cfg.setdefault("window", {})["window_size_s"] = self._normalize_window_size_s(window_s)
                        run_window_cfg["data_augmentation"] = deepcopy(data_augmentation)
                        
                        window_ms = window_ms_from_cfg(run_window_cfg)
                        model_name_id = f"w{window_ms}ms"

                        print(f"\n[{run_label}] {subject} | {condition} | window: {model_name_id}")

                        # 2. Loop on the number of sessions
                        for n_sessions in range(1, max_sessions_available + 1):
                            selected_sessions = sessions[:n_sessions]
                            artifacts_n = self.artifacts_dir / run_label / f"{n_sessions}_sess"
                            extraction_success = self._run_windowing(
                                run_window_cfg, combo_data_root, subject, condition, window_s, data_augmentation
                            )
                            if not extraction_success:
                                print(f"[SKIP] No windows generated for {run_label}.")
                                continue

                            artifacts_n.mkdir(parents=True, exist_ok=True)

                            cfg_run = deepcopy(self.base_cfg)
                            cfg_run.setdefault("data", {})
                            cfg_run["data_augmentation"] = deepcopy(data_augmentation)
                            cfg_run.setdefault("window", {})["window_size_s"] = self._normalize_window_size_s(window_s)

                            # 3. Loop on the experiments
                            for current_exp in self.experiments:
                                res = self._evaluate_single_experiment(
                                    current_exp, cfg_run, combo_data_root, artifacts_n,
                                    subject, condition, model_name_id, selected_sessions,
                                    data_augmentation, run_label, window_s, safe_num_strides, n_sessions
                                )
                                if res:
                                    self.ablation_results.append(res)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=Path, required=True)
    ap.add_argument("--model_config", type=Path, required=True)
    ap.add_argument("--window_config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, required=True)
    ap.add_argument("--experiment", nargs="+", choices=["global", "inter_session"], default=["global", "inter_session"])
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    ap.add_argument("--aug_windows_s", nargs="*", type=float, default=[1.4])
    ap.add_argument("--stride_ms", nargs="*", type=int, default=[10])
    ap.add_argument("--num_strides", nargs="*", type=int, default=[2, 5, 10])
    args = ap.parse_args()

    trainer = Data_Augmentation_Ablation_Trainer(
        base_config=yaml.safe_load(args.base_config.read_text()),
        model_config=yaml.safe_load(args.model_config.read_text()),
        window_config=yaml.safe_load(args.window_config.read_text()),
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        subjects=args.subjects,
        conditions=args.conditions,
        experiments=args.experiment,
        windows_s=args.aug_windows_s,
        stride_ms=args.stride_ms,
        num_strides=args.num_strides,
    )
    trainer.main()