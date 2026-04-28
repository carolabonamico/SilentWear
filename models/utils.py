# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Utils function for models
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from models.models_factory import ModelSpec, build_model_from_spec
import torch
from utils.I_data_preparation.experimental_config import FS


def compute_metrics(y_true, y_pred):
    """
    Docstring for compute_metrics

    :param y_true: true labels
    :param y_pred: predicted labels

    Returns metrics,y_true, y_pred
    """
    # ===== Metrics summary =====
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Add also weighted metrics to take into account class imbalace
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }

    print("\n=== Test Metrics ===")
    print(f"{ 'Accuracy':<15}: UNBALANCED {acc:6.2f}  - BALANCED {balanced_acc:6.2f}")
    print(f"{ 'Precision':<15}: MACRO      {precision_macro:6.2f}  - WEIGHTED {precision_weighted:6.2f}")
    print(f"{ 'Recall':<15}: MACRO      {recall_macro:6.2f}  - WEIGHTED {recall_weighted:6.2f}")
    print(f"{ 'F1-score':<15}: MACRO      {f1_macro:6.2f}  - WEIGHTED {f1_weighted:6.2f}")

    # print(cm)

    return metrics, y_true, y_pred


import torch.nn as nn


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def check_weights_updated(before_state_dict, model_after):
    """
    Returns True if at least one parameter tensor differs after loading.
    """
    after_sd = model_after.state_dict()
    changed = False

    for k, v_before in before_state_dict.items():
        if k not in after_sd:
            continue
        v_after = after_sd[k]

        # only compare tensors
        if torch.is_tensor(v_before) and torch.is_tensor(v_after):
            if not torch.equal(v_before, v_after):
                changed = True
                break

    return changed


def resolve_num_classes_from_cfg(
    base_cfg: dict, model_cfg: dict, train_label_map: dict | None = None
) -> int:
    """Resolve output classes from config."""
    include_rest = bool(base_cfg["experiment"].get("include_rest", False))
    num_classes = 9 if include_rest else 8

    model_section = model_cfg.get("model", {}) or {}
    kwargs = model_section.get("kwargs", {}) or {}
    train_cfg = kwargs.get("train_cfg", {}) or {}
    loss_name = str(train_cfg.get("loss_name", "")).lower().strip()

    if model_section.get("kind") == "dl" and loss_name == "ctc":
        # CTC output size is based on token vocabulary.
        from utils.I_data_preparation.ctc_text_mapper import CTCTextMapper, DEFAULT_BLANK_ID

        ctc_cfg = train_cfg.get("ctc")
        if not isinstance(ctc_cfg, dict):
            raise KeyError("For loss_name='ctc', model.kwargs.train_cfg.ctc must be provided.")

        lexicon_path = ctc_cfg.get("lexicon_path")
        if not lexicon_path:
            raise ValueError(
                "For loss_name='ctc', provide model.kwargs.train_cfg.ctc.lexicon_path."
            )

        mapper = CTCTextMapper(
            lexicon_path=lexicon_path,
            train_label_map=train_label_map or {},
            blank_id=ctc_cfg.get("blank_id", DEFAULT_BLANK_ID),
        )
        num_classes = len(mapper.char_to_int)
        print("CTC token vocab size (without blank):", num_classes)

    return num_classes


def load_pretrained_model(base_cfg, model_cfg, pretrained_model_path):
    num_classes = resolve_num_classes_from_cfg(base_cfg, model_cfg)

    spec = ModelSpec(
        kind=model_cfg["model"]["kind"],
        name=model_cfg["model"]["name"],
        kwargs=model_cfg["model"]["kwargs"],
    )

    ctx = {
        "num_channels": 14,
        "num_samples": int(base_cfg["window"]["window_size_s"] * FS),
        "num_classes": num_classes,
    }

    model = build_model_from_spec(spec, ctx)

    # snapshot BEFORE
    before_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if str(pretrained_model_path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(pretrained_model_path)
    else:
        cpt = torch.load(pretrained_model_path, map_location="cpu")
        state_dict = cpt.get("model_state_dict", cpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    if check_weights_updated(before_sd, model):
        return model
    else:
        print("No weights changed after load — check checkpoint keys / strictness.")
        return None
