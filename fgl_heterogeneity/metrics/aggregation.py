"""Aggregation helpers for federation-level heterogeneity summaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np


def aggregate_pairwise_metrics(pairwise_matrix: np.ndarray) -> Dict[str, Any]:
    """Summarize a symmetric pairwise metric matrix at the federation level."""
    if pairwise_matrix.ndim != 2 or pairwise_matrix.shape[0] != pairwise_matrix.shape[1]:
        raise ValueError("pairwise_matrix must be square")

    K = pairwise_matrix.shape[0]
    if K == 0:
        raise ValueError("pairwise_matrix must contain at least one client")
    if K == 1:
        return {
            "mean_pairwise": 0.0,
            "max_pairwise": 0.0,
            "per_client_mean": [0.0],
            "variance": 0.0,
        }

    triu_vals = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    per_client = []
    for i in range(K):
        others = [pairwise_matrix[i, j] for j in range(K) if j != i]
        per_client.append(float(np.mean(others)) if others else 0.0)

    return {
        "mean_pairwise": float(np.mean(triu_vals)),
        "max_pairwise": float(np.max(triu_vals)),
        "per_client_mean": per_client,
        "variance": float(np.var(triu_vals)) if len(triu_vals) > 0 else 0.0,
    }


def compute_full_heterogeneity_profile(
    label_counts: Optional[List[np.ndarray]],
    embeddings: Optional[List[np.ndarray]],
    graphs: Optional[List[Any]],
    node_labels: Optional[List[Dict[int, int]]],
    node_sets: Optional[List[Set[int]]],
    global_adj: Optional[Dict[int, Set[int]]],
    sample_sizes: List[int],
    subspace_rank: int = 10,
) -> Dict[str, Any]:
    """Compute the full recommended metric suite where inputs are available."""
    from .label_metrics import label_distribution_divergence
    from .overlap_metrics import missing_neighbor_ratio, neighbor_overlap_index
    from .quantity_metrics import quantity_imbalance_index
    from .representation_metrics import (
        embedding_centroid_divergence,
        principal_subspace_divergence,
    )
    from .topology_metrics import homophily_gap, topological_divergence

    profile: Dict[str, Any] = {}

    if label_counts is not None:
        label_res = label_distribution_divergence(label_counts)
        profile["label_jsd"] = aggregate_pairwise_metrics(label_res["pairwise_jsd"])
        profile["lpv"] = float(label_res["lpv"])

    if embeddings is not None:
        ecd_res = embedding_centroid_divergence(embeddings)
        profile["ecd"] = aggregate_pairwise_metrics(ecd_res["pairwise_ecd"])

        psd_res = principal_subspace_divergence(embeddings, rank=subspace_rank)
        profile["psd"] = aggregate_pairwise_metrics(psd_res["pairwise_psd"])

    if graphs is not None and node_labels is not None:
        hg_res = homophily_gap(graphs, node_labels)
        profile["homophily_gap"] = aggregate_pairwise_metrics(hg_res["pairwise_hg"])
        profile["homophily_variance"] = float(hg_res["variance"])

    if graphs is not None:
        td_res = topological_divergence(graphs, metric="degree")
        profile["topo_divergence"] = aggregate_pairwise_metrics(td_res["pairwise_td"])

    if node_sets is not None:
        noi_res = neighbor_overlap_index(node_sets)
        profile["noi"] = float(noi_res["mean_overlap"])
        profile["pairwise_noi"] = noi_res["pairwise_noi"]
        if global_adj is not None:
            mnr_list = missing_neighbor_ratio(node_sets, global_adj)
            profile["mnr_per_client"] = [float(x) for x in mnr_list]
            profile["mnr_mean"] = float(np.mean(mnr_list)) if mnr_list else 0.0

    qii_res = quantity_imbalance_index(sample_sizes)
    profile["qii"] = float(qii_res["qii"])

    profile["domain_shift_profile"] = {
        "mean_jsd": profile.get("label_jsd", {}).get("mean_pairwise", 0.0),
        "mean_ecd": profile.get("ecd", {}).get("mean_pairwise", 0.0),
        "mean_psd": profile.get("psd", {}).get("mean_pairwise", 0.0),
        "mean_hg": profile.get("homophily_gap", {}).get("mean_pairwise", 0.0),
        "mean_td": profile.get("topo_divergence", {}).get("mean_pairwise", 0.0),
        "mean_mnr": profile.get("mnr_mean", 0.0),
        "qii": profile["qii"],
    }
    return profile
