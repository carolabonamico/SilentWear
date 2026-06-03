#!/usr/bin/env python3

"""
Session-count ablation for offline experiments.

This script runs `global` or `inter_session` experiments by progressively increasing the
number of included sessions from 1 to N available.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yaml
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.I_global_models import Global_Model_Trainer
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import reset_all_seeds
from utils.I_data_preparation.read_bio_file import parse_bio_filename
from utils.general_utils import load_all_h5files_from_folder, window_ms_from_cfg

DATA_DIR = REPO_ROOT / "data" / "data_3_subjects_4_sessions_in_order_ACTUAL"
ARTIFACTS_DIR = REPO_ROOT / "artifacts_ablation" / "session_count_ablation"
DEFAULT_WINDOWS_S = [1.4]
DEFAULT_SUBJECTS = ["S01", "S02", "S03", "S04"]
DEFAULT_EXPERIMENTS = ["global", "inter_session"]
DEFAULT_CONDITIONS = ["silent", "vocalized"]


class Session_Count_Ablation_Trainer:
    def __init__(
        self,
        base_config: dict,
        model_config: dict,
        data_dir: Path,
        artifacts_dir: Path,
        subjects: List[str],
        conditions: List[str],
        experiments: List[str],
        min_sessions: int = 1,
        windows_s: Optional[List[float]] = None,
    ) -> None:
        
        self.base_cfg = deepcopy(base_config)
        self.model_cfg = deepcopy(model_config)
        
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        
        self.subjects = subjects
        self.conditions = conditions
        self.experiments = experiments if experiments else DEFAULT_EXPERIMENTS
        
        self.min_sessions = min_sessions
        self.windows_s = windows_s if windows_s else DEFAULT_WINDOWS_S
        
        self.ablation_results: List[Dict[str, Any]] = []

    def _discover_sessions(self, subject: str, condition: str) -> List[int]:
        """Discover available session IDs for a given subject and condition."""
        raw_dir = self.data_dir / "raw" / subject / condition
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

        sessions = set()
        for bio_path in sorted(raw_dir.glob("*.bio")):
            parsed = parse_bio_filename(bio_path)
            if parsed is None:
                continue
            sess_id, _, _ = parsed
            sessions.add(int(sess_id))

        sessions_sorted = sorted(sessions)
        if not sessions_sorted:
            raise ValueError(f"No valid BIO session files found in: {raw_dir}")

        return sessions_sorted

    def _load_windows_df(self, data_dirs: List[Path]) -> pd.DataFrame:
        """Load and concatenate windows DataFrames from multiple directories."""
        frames = [
            load_all_h5files_from_folder(d, key="wins_feats", print_statistics=False)
            for d in data_dirs
        ]
        return pd.concat(frames, ignore_index=True).reset_index(drop=True)

    def _set_common_run_cfg(
        self,
        base_cfg: dict,
        artifacts_root: Path,
        subject: str,
        condition: str,
        model_name_id: str,
        selected_sessions: List[int]
    ) -> dict:
        """Set common configuration for both global and inter-session experiments."""
        cfg = deepcopy(base_cfg)
        cfg.setdefault("data", {})
        cfg.setdefault("experiment", {})

        cfg["data"]["data_directory"] = str(self.data_dir)
        cfg["data"]["models_main_directory"] = str(artifacts_root)
        cfg["data"]["subject_id"] = str(subject)
        cfg["condition"] = str(condition)
        cfg["model_name_id"] = str(model_name_id)

        cfg["experiment"]["sessions_ablation"] = {
            "num_sessions": int(len(selected_sessions)),
            "session_ids": [int(s) for s in selected_sessions],
        }

        return cfg

    def _run_global_model(
        self,
        cfg_run: dict,
        artifacts_n: Path,
        subject: str,
        condition: str,
        model_name_id: str,
        selected_sessions: List[int]
    ) -> Path:
        """Run the global experiment for the given subject, condition, and selected sessions."""
        cfg = self._set_common_run_cfg(cfg_run, artifacts_n, subject, condition, model_name_id, selected_sessions)
        trainer = Global_Model_Trainer(base_config=cfg, model_config=deepcopy(self.model_cfg))
        df_full = self._load_windows_df(trainer.data_dire_proc)
        trainer.df = df_full[df_full["session_id"].isin(selected_sessions)].copy().reset_index(drop=True)

        if trainer.df.empty:
            raise ValueError(f"No rows left for GLOBAL after filtering sessions={selected_sessions}")

        trainer._save_run_cfg()
        trainer.run_cv()
        pd.DataFrame(trainer.cv_summaries).to_csv(trainer.model_dire / "cv_summary.csv", index=False)
        return trainer.model_dire

    def _run_inter_session_model(
        self,
        cfg_run: dict,
        artifacts_n: Path,
        subject: str,
        condition: str,
        model_name_id: str,
        selected_sessions: List[int]
    ) -> Path:
        """Run the inter-session experiment for the given subject, condition, and selected sessions."""
        cfg = self._set_common_run_cfg(cfg_run, artifacts_n, subject, condition, model_name_id, selected_sessions)
        trainer = Inter_Session_Model_Trainer(
            base_config=cfg, model_config=deepcopy(self.model_cfg), experiment_subdir="inter_session"
        )

        df_full = self._load_windows_df(trainer.data_dire_proc)
        trainer.df = df_full[df_full["session_id"].isin(selected_sessions)].copy().reset_index(drop=True)

        if len(np.sort(trainer.df["session_id"].unique())) < 2:
            raise ValueError("inter_session requires at least 2 sessions.")

        trainer._save_run_cfg()
        val_size = float(cfg.get("cv", {}).get("val_size", 0.3))
        seed = int(cfg.get("experiment", {}).get("seed", 0))
        trainer.run_inter_session_cv(val_size=val_size, seed=seed)
        pd.DataFrame(trainer.cv_summaries).to_csv(trainer.model_dire / "cv_summary.csv", index=False)
        return trainer.model_dire

    def _evaluate_single_experiment(
        self,
        current_exp: str,
        cfg_run: dict,
        artifacts_n: Path,
        subject: str,
        condition: str,
        model_name_id: str,
        selected_sessions: List[int],
        window_s: float, 
        n_sessions: int
    ) -> Optional[dict]:
        """Run a single experiment (global or inter-session) for the given subject, condition, and number of sessions."""
        try:
            reset_all_seeds()

            if current_exp == "global":
                out_dir = self._run_global_model(
                    cfg_run, artifacts_n, subject, condition, model_name_id, selected_sessions
                )
            elif current_exp == "inter_session":
                if len(selected_sessions) < 2:
                    print("[SKIP][inter_session] needs >=2 sessions")
                    return None
                out_dir = self._run_inter_session_model(
                    cfg_run, artifacts_n, subject, condition, model_name_id, selected_sessions
                )
            else:
                return None

            print(f"[DONE][{current_exp}] {out_dir}")
            return {"experiment": current_exp, "n_sessions": n_sessions}

        except ValueError as e:
            print(f"[SKIP] Un-trainable for {n_sessions} sessions - {current_exp}: {e}")
            return None

    def main(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        max_sessions_global = max(
            len(self._discover_sessions(subj, cond))
            for subj in self.subjects
            for cond in self.conditions
        )

        if max_sessions_global == 0:
            print("[ERROR] No sessions found for any subject/condition.")
            return

        print(f"\n[ABLATION] Running session-count ablation from {self.min_sessions} to {max_sessions_global} sessions.")

        # 1. Loop on the number of sessions
        for n_sessions in range(self.min_sessions, max_sessions_global + 1):
            print(f"\n{'='*90}")
            print(f"[STARTING PHASE] -> {n_sessions} SESSION(S)")
            print(f"{'='*90}")

            # 2. Loop on the experiments
            for current_exp in self.experiments:

                # 3. Loop on the window sizes (if applicable)
                for window_s in self.windows_s:
                    cfg_base = deepcopy(self.base_cfg)
                    cfg_base.setdefault("window", {})["window_size_s"] = float(window_s)
                    window_ms = window_ms_from_cfg(cfg_base)
                    model_name_id = f"w{window_ms}ms"

                    print(f"\n  [{current_exp.upper()}] | Window: {model_name_id}")

                    # 4. Loop on subject-condition pairs
                    for subject in self.subjects:
                        for condition in self.conditions:
                            sessions = self._discover_sessions(subject, condition)
                            if n_sessions > len(sessions):
                                continue

                            print(f"-> {subject} | {condition}")
                            selected_sessions = sessions[:n_sessions]
                            artifacts_n = self.artifacts_dir / f"{n_sessions}_sess"
                            artifacts_n.mkdir(parents=True, exist_ok=True)

                            res = self._evaluate_single_experiment(
                                current_exp, cfg_base, artifacts_n, subject, condition,
                                model_name_id, selected_sessions, window_s, n_sessions
                            )
                            if res:
                                self.ablation_results.append(res)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=Path, required=True)
    ap.add_argument("--model_config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--artifacts_dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--experiment", nargs="+", choices=DEFAULT_EXPERIMENTS, default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--min_sessions", type=int, default=1)
    ap.add_argument("--windows_s", nargs="*", type=float, default=None)
    args = ap.parse_args()

    trainer = Session_Count_Ablation_Trainer(
        base_config=yaml.safe_load(args.base_config.read_text()),
        model_config=yaml.safe_load(args.model_config.read_text()),
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        subjects=args.subjects,
        conditions=args.conditions,
        experiments=args.experiment,
        min_sessions=args.min_sessions,
        windows_s=args.windows_s,
    )
    trainer.main()