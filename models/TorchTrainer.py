# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Main Trainer for Deep Learning Models (pytorch based)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from models.seeds import *
from models.utils import compute_metrics
from typing import Dict, List, Optional
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import hashlib
from models.strategies import TaskStrategy

EARLY_STOP_PATIENCE = 5  # default value if not specified in train_cfg


class TorchTrainer:
    def __init__(self, estimator, df_train, df_val, df_test, train_cfg, label_col,
                 strategy: TaskStrategy, train_label_map: Optional[Dict[int, str]] = None):
        self.model = estimator
        self.df_train = df_train
        self.train_loader = None
        self.df_val = df_val
        self.valoader = None
        self.df_test = df_test
        self.test_loader = None
        self.train_cfg = train_cfg
        self.label_col = label_col
        self.train_label_map = train_label_map or {}

        if strategy is None:
            raise ValueError("A TaskStrategy (e.g., CrossEntropyStrategy or CTCStrategy) must be explicitly provided.")
        self.strategy = strategy

    def create_dataloader_from_df(
        self, df, batch_size=32, shuffle=False, num_workers=0  # we shuffle outside
    ):
        """
        Function to create a dataloader from a given df.
        """
        if df is None or df.empty is None:
            return None
        X_df = df.drop(columns=self.label_col)
        print(X_df.columns)

        # (N, T) per channel, then stack to (N, C, T)
        X_np = np.stack([np.stack(X_df[col].to_numpy()) for col in X_df.columns], axis=1).astype(
            np.float32
        )
        # X_np shape: (N, C, T)

        y_np = df[self.label_col].to_numpy()

        X_torch = torch.from_numpy(X_np)  # (N, C, T)
        print("Tensor data with shape", X_torch.shape)
        y_torch = torch.from_numpy(y_np)

        # Build dataset and dataloader
        dataset = TensorDataset(X_torch, y_torch)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

        return dataloader

    def build_scheduler(self, optimizer, scheduler_cfg, num_epochs):
        if not scheduler_cfg:
            return None

        name = scheduler_cfg["name"]
        print("Building scheduler:", name)
        if name in ("", "none", "null"):
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg.get("T_max", num_epochs)),
                eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
            )
        elif name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "min"),
                factor=float(scheduler_cfg.get("factor", 0.1)),
                patience=int(scheduler_cfg.get("patience", 10)),
            )

        raise ValueError(f"Unknown scheduler: {name}")

    def train_loop(
        self,
        model,
        trainloader,
        valoader,
        train_cfg,
        save_path,
    ):

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Running on cuda")
            print(torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("Running on CPU")

        # ---- Read config ----
        num_epochs = int(train_cfg.get("num_epochs", 50))

        # optimizer_cfg = train_cfg.get("optimizer", None) or {"name": "adam", "lr": 1e-3}
        optimizer_cfg = (
            train_cfg.get("optimizer_cfg", None)
            or train_cfg.get("optimizer", None)
            or {"name": "adam", "lr": 1e-3}
        )

        opt_name = str(optimizer_cfg.get("name", "adam")).lower()

        lr = float(optimizer_cfg["lr"])
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        # betas = optimizer_cfg.get("betas", (0.9, 0.999))

        print("Model will be trained for:", num_epochs, "epochs")
        early_stop_patience = train_cfg.get("early_stop_patience", EARLY_STOP_PATIENCE)
        print("Early stop patienence set to:", early_stop_patience)
        print("Set optimizer", opt_name, "|lr:", lr, "|wd:", weight_decay)
        scheduler_cfg = train_cfg.get("scheduler", None)
        print("Set scheduler:", scheduler_cfg)
        grad_clip_norm = train_cfg.get("grad_clip_norm", None)
        if grad_clip_norm is not None:
            print("Set gradient clipping max_norm:", float(grad_clip_norm))

        # ----- Optimizer -----
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # ----- Scheduler (optional) -----
        scheduler = self.build_scheduler(optimizer, scheduler_cfg, num_epochs)

        # Warmup configuration: linearly scale LR during the first N epochs.
        # If `warmup_epochs` is set in `train_cfg`, we interpolate from 0->base_lr.
        warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        if warmup_epochs > 0:
            print(f"Warmup for {warmup_epochs} epochs; base_lrs: {base_lrs}")

        # -------- Zero-epoch (before training) evaluation --------
        model.eval()
        with torch.no_grad():
            # Train accuracy before training
            train_preds, train_tgts = [], []
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device).long()
                out = model(x)
                preds = self.strategy.predict_labels(out)
                # ensure list-like for extend
                train_preds.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                train_tgts.extend(y.cpu().numpy().tolist())
            train_acc_0 = np.mean(np.array(train_preds) == np.array(train_tgts))

            # Validation accuracy before training
            val_preds, val_tgts = [], []
            for x, y in valoader:
                x = x.to(device)
                y = y.to(device).long()
                out = model(x)
                preds = self.strategy.predict_labels(out)
                val_preds.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                val_tgts.extend(y.cpu().numpy().tolist())
            val_acc_0 = np.mean(np.array(val_preds) == np.array(val_tgts))

        print(f"PRE-TRAIN | TRAIN ACC: {train_acc_0:.3f} | VAL ACC: {val_acc_0:.3f}")

        # -------- Real Training Starts --------
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None

        patience = 0

        model.to(device)
        train_accs = []
        val_accs = []
        for epoch in range(num_epochs):
            # -------- Train --------
            # Warmup: scale learning rate linearly during initial epochs
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_scale = float(epoch + 1) / float(warmup_epochs)
                for i, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = base_lrs[i] * warmup_scale
            model.train()
            running_loss_train = 0.0
            train_batches = 0
            train_predictions = []
            train_targets = []

            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device).long()

                optimizer.zero_grad()
                outputs = model(x)
                loss = self.strategy.compute_loss(outputs, y, device)
                self.strategy.backward(loss)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()

                running_loss_train += loss.item()
                train_batches += 1
                preds = self.strategy.predict_labels(outputs)
                train_predictions.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                train_targets.extend(y.cpu().numpy().tolist())
            train_accuracy = np.mean(np.array(train_predictions) == np.array(train_targets))
            avg_train_loss = running_loss_train / max(1, train_batches)
            train_accs.append(train_accuracy)

            # -------- Validation --------
            model.eval()
            running_loss_val = 0.0
            val_batches = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for x, y in valoader:
                    x = x.to(device)
                    y = y.to(device).long()

                    outputs = model(x)
                    loss = self.strategy.compute_loss(outputs, y, device)

                    running_loss_val += loss.item()
                    val_batches += 1

                    # Collect predictions and targets for later analysis
                    preds = self.strategy.predict_labels(outputs)
                    val_predictions.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
                    val_targets.extend(y.cpu().numpy().tolist())
            # compute accuracy on the validation set
            val_accuracy = np.mean(np.array(val_predictions) == np.array(val_targets))
            val_accs.append(val_accuracy)

            avg_val_loss = running_loss_val / max(1, val_batches)

            # ----- Scheduler step (safe fallback) -----
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # print("stepping scheduler ReduceLROnPlateau")
                    current_lr = scheduler.get_last_lr()[0]
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Early stopping bookkeeping (UNCHANGED)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    # optional: store effective optimizer/scheduler settings used
                    "optimizer_cfg": optimizer_cfg,
                    "scheduler_cfg": scheduler_cfg,
                }
                patience = 0
            else:
                patience += 1
                print(f"Consecutive epochs without improvement: {patience}")
                if patience >= early_stop_patience:
                    print("Hit early stopping.")
                    break
            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                print(
                    f"{epoch} TRAIN loss: {avg_train_loss:.3f} | VAL loss: {avg_val_loss:.3f} | TRAIN ACC: {train_accuracy:.3f} | VAL ACC: {val_accuracy:.3f} | LR: {current_lr:.2e}"
                )
            else:
                print(
                    f"{epoch} TRAIN loss: {avg_train_loss:.3f} | VAL loss: {avg_val_loss:.3f} | TRAIN ACC: {train_accuracy:.3f} | VAL ACC: {val_accuracy:.3f}"
                )
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

        epochs_ran = len(train_losses)  # how many epochs actually executed
        best_epoch_1based = (best_state["epoch"] + 1) if best_state is not None else None

        # Save best model and loss history (UNCHANGED)
        if save_path is not None and best_state is not None:
            best_state.update(
                {
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                    "train_acc": train_accs,
                    "val_acc": val_accs,
                    "best_val_loss": best_val_loss,
                    "requested_num_epochs": int(num_epochs),
                    "epochs_ran": int(epochs_ran),
                    "best_epoch": int(best_epoch_1based),
                    "early_stop_patience": int(early_stop_patience),
                }
            )
            torch.save(best_state, save_path)

        # Optionally restore best model weights before returning (UNCHANGED)
        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])

        return model

    def fit(self, save_model_path: Optional[Path] = None):
        """
        Train Pytorch model on features X and labels y.
        """

        self.train_loader = self.create_dataloader_from_df(self.df_train)
        self.valoader = self.create_dataloader_from_df(self.df_val)
        self.test_loader = self.create_dataloader_from_df(self.df_test)
        # Check that splits are truly different
        self.check_data_splits()

        # Fit estimator
        if save_model_path is not None:
            save_model_path = Path(save_model_path)
            model_path = (
                save_model_path
                if save_model_path.suffix == ".pt"
                else save_model_path.with_suffix(".pt")
            )
        else:
            model_path = None

        self.model = self.train_loop(
            self.model, self.train_loader, self.valoader, self.train_cfg, model_path
        )
        return self.model

    def evaluate(self, test_loader=None):
        """
        Evaluates the model on the test set and returns metrics, true labels, and predicted labels.
        If test_loader is provided, it will be used instead of the default test_loader created from df_test.
        """
        loader = test_loader if test_loader is not None else self.test_loader
        return evaluate_model(self.model, loader, self.strategy)

    def check_data_splits(self):
        """
        Verify that train/val/test splits are disjoint.
        Detects overlap by index and by hashing sample content.
        Safe if one of the splits is None or empty.
        """
        # print("Checking data split integrity...")

        splits = {
            "TRAIN": getattr(self, "df_train", None),
            "VAL": getattr(self, "df_val", None),
            "TEST": getattr(self, "df_test", None),
        }

        # Keep only non-empty DataFrames
        valid = {}
        for name, df in splits.items():
            if df is None:
                # print(f" - {name}: None (skipping)")
                continue
            if df.empty:
                # print(f" - {name}: empty (skipping)")
                continue
            valid[name] = df
            # print(f" - {name}: {len(df)} samples")

        if len(valid) < 2:
            print("Not enough splits present to compare overlap (need at least 2).")
            print("Split integrity check complete.\n")
            return

        # ------------------------------------------------------------
        # 1) Check index overlap (fast)
        # ------------------------------------------------------------
        idx_sets = {name: set(df.index) for name, df in valid.items()}

        names = list(idx_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = idx_sets[a].intersection(idx_sets[b])
                if overlap:
                    print(f"Overlap {a}–{b} (index): {len(overlap)} samples")
                else:
                    continue
                    # print(f"No index overlap detected for {a}–{b}.")

        # ------------------------------------------------------------
        # 2) Check overlap by sample content (robust)
        # ------------------------------------------------------------
        def hash_df(df):
            """
            Create a stable hash per row even if cells contain numpy arrays.
            """

            def row_fingerprint(row) -> str:
                h = hashlib.blake2b(digest_size=16)
                for v in row:
                    if isinstance(v, np.ndarray):
                        h.update(str(v.shape).encode())
                        h.update(str(v.dtype).encode())
                        h.update(v.tobytes())
                    elif isinstance(v, (list, tuple)):
                        arr = np.asarray(v)
                        h.update(str(arr.shape).encode())
                        h.update(str(arr.dtype).encode())
                        h.update(arr.tobytes())
                    else:
                        h.update(str(v).encode())
                return h.hexdigest()

            return df.apply(row_fingerprint, axis=1)

        hash_sets = {name: set(hash_df(df)) for name, df in valid.items()}

        names = list(hash_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = hash_sets[a].intersection(hash_sets[b])
                if overlap:
                    print(f"Content overlap {a}–{b}: {len(overlap)} samples")
                else:
                    # print(f"No content overlap detected for {a}–{b}.")
                    continue

        print("Split integrity check complete.\n")


################################################### Standalone functions ########################


def evaluate_model(model, test_loader, strategy: TaskStrategy):
    """
    Compute predictions and metrics on the test set using the provided model, test_loader, and strategy.
    """
    if strategy is None:
        raise ValueError("A TaskStrategy must be explicitly provided for evaluation.")

    device = next(model.parameters()).device
    model.eval()

    all_targets = []
    all_preds = []
    all_logits = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            outputs = model(inputs)

            # Use strategy to obtain final predicted labels (handles CE and CTC)
            preds_np = strategy.predict_labels(outputs)

            # Normalize to numpy arrays
            if isinstance(preds_np, np.ndarray):
                batch_preds = preds_np
            else:
                try:
                    batch_preds = np.asarray(preds_np)
                except Exception:
                    # Fall back to converting torch tensors
                    batch_preds = preds_np.detach().cpu().numpy()

            all_preds.append(batch_preds)
            all_targets.append(targets.cpu().numpy())
            try:
                all_logits.append(outputs.cpu().numpy())
            except Exception:
                # outputs may be a tuple/list; ignore logits if not convertible
                pass

    # Concatenate batches along first axis
    if len(all_targets) == 0:
        return None, None, None

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    metrics, y_true, y_pred = compute_metrics(y_true, y_pred)

    return metrics, y_true, y_pred
