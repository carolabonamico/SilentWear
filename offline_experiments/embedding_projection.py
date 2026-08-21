# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Analyze learned model embeddings.

Embeddings are extracted from a trained SpeechNet at two points - before the
BiLSTM layer and before the final fully-connected layer - and:
  1. projected to 2D with UMAP / t-SNE / PCA (single global fit, coloured by
     class) for VISUALIZATION of cluster structure;
  2. scored with quantitative separability metrics computed on the FULL
     embeddings (silhouette, Davies-Bouldin, intra/inter-class distance,
     separation ratio) to actually QUANTIFY how far apart the class clusters
     are - and to compare layers.

Note: UMAP/t-SNE 2D coordinates do not preserve distances, so distance
evaluation is done on the full embeddings, never on the 2D projection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


def collect_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run the model over a (non-shuffled) loader and stack per-sample embeddings."""
    model.eval()
    collected: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            for name, emb in model.extract_embeddings(x).items():
                collected.setdefault(name, []).append(emb.cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def compute_separability_metrics(X: np.ndarray, labels: np.ndarray) -> Optional[dict]:
    """Quantify how well classes separate in a (full-dimensional) embedding space.

    Distances are measured on the raw embeddings, NOT on the 2D UMAP/t-SNE/PCA
    output (those are for visualization and do not preserve distances). The
    dimensionless metrics (silhouette, davies_bouldin, separation_ratio) are
    comparable across layers of different dimensionality; the absolute
    intra/inter distances are not.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n = len(X)
    if len(classes) < 2 or n <= len(classes):
        return None

    centroids = np.stack([X[labels == c].mean(axis=0) for c in classes])
    intra = float(
        np.mean(
            [np.linalg.norm(X[labels == c] - centroids[i], axis=1).mean() for i, c in enumerate(classes)]
        )
    )
    inter = float(
        np.mean(
            [
                np.linalg.norm(centroids[i] - centroids[j])
                for i in range(len(classes))
                for j in range(i + 1, len(classes))
            ]
        )
    )
    sample_size = 2000 if n > 2000 else None
    return {
        "n_samples": int(n),
        "n_classes": int(len(classes)),
        "silhouette": float(silhouette_score(X, labels, sample_size=sample_size, random_state=42)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "intra_class_dist": intra,
        "inter_class_dist": inter,
        "separation_ratio": float(inter / intra) if intra > 0 else float("inf"),
    }


def _build_extractor(method: str, out_dir: Path, subject_id: str):
    from utils.II_feature_extraction.ProjectionExtractor import (
        UMAP_Projection_Extractor,
        UMAPConfig,
        TSNE_Projection_Extractor,
        TSNEConfig,
        PCA_Projection_Extractor,
        PCAConfig,
    )

    if method == "umap":
        return UMAP_Projection_Extractor(UMAPConfig(max_points=5000), out_dir, subject_id)
    if method == "tsne":
        return TSNE_Projection_Extractor(TSNEConfig(max_points=5000), out_dir, subject_id)
    if method == "pca":
        return PCA_Projection_Extractor(PCAConfig(max_points=5000), out_dir, subject_id)
    raise ValueError(f"Unknown projection method: {method}")


def run_embedding_projection(
    model_master,
    out_dir: Path,
    methods: Sequence[str],
    layers: Sequence[str],
    condition: str,
) -> None:
    """Extract embeddings from the trained model and project them with each method.

    Plots are written to ``out_dir/<layer>/<method>/`` and coloured by class label.
    Only applies to deep-learning models that expose ``extract_embeddings`` (SpeechNet).
    """
    if getattr(model_master, "kind", None) != "dl":
        return
    model = model_master.model
    if not hasattr(model, "extract_embeddings"):
        print("[EMBEDDING] Model has no extract_embeddings; skipping projection.")
        return
    if not methods or not layers:
        return

    df_test = model_master.df_test
    if df_test is None or df_test.empty:
        print("[EMBEDDING] Empty test set; skipping projection.")
        return
    df_test = df_test.reset_index(drop=True)

    data_cols = model_master.data_col_to_consider + ["Label_train"]
    loader = model_master.trainer_manager.create_dataloader_from_df(
        df_test[data_cols], shuffle=False
    )

    device = next(model.parameters()).device
    embeddings = collect_embeddings(model, loader, device)

    subject_id = str(model_master.base_config.get("data", {}).get("subject_id", "subject"))
    out_dir = Path(out_dir)
    labels = df_test["Label_train"].to_numpy()
    metric_rows = []

    for layer in layers:
        if layer not in embeddings:
            continue
        X = embeddings[layer]

        # Quantitative separability on the FULL embeddings (comparable across layers).
        metrics = compute_separability_metrics(X, labels)
        if metrics is not None:
            metric_rows.append({"layer": layer, **metrics})

        emb_cols = [f"emb_{j}" for j in range(X.shape[1])]
        df_emb = df_test.copy()
        df_emb[emb_cols] = X

        for method in methods:
            print(f"[EMBEDDING] {layer} | {method.upper()} | {subject_id} | {condition}")
            extractor = _build_extractor(method, out_dir / layer / method, subject_id)
            # Single global fit over all test embeddings, coloured by class label.
            extractor.plot_single(df_emb, emb_cols, condition=condition, show=False)

    if metric_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metric_rows).to_csv(out_dir / "separability_metrics.csv", index=False)
        print(f"[EMBEDDING] Saved separability metrics: {out_dir / 'separability_metrics.csv'}")


def maybe_run_embedding_projection(
    model_master,
    base_config: dict,
    out_dir: Path,
    condition: str,
) -> None:
    """Run embedding projection when enabled via experiment.embedding_projection."""
    cfg = base_config.get("experiment", {}).get("embedding_projection", {}) or {}
    if not cfg.get("enabled"):
        return
    methods = cfg.get("methods", ["umap", "tsne", "pca"])
    layers = cfg.get("layers", ["pre_bilstm", "pre_fc"])
    run_embedding_projection(model_master, out_dir, methods, layers, condition)
