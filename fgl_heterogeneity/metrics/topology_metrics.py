"""Topology-aware heterogeneity metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from .label_metrics import jensen_shannon_divergence


def compute_homophily(graph: nx.Graph, labels: Dict[int, int]) -> float:
    """Compute edge homophily h_k = |{(u,v): y_u=y_v}| / |E_k|."""
    if graph.number_of_edges() == 0:
        return 0.0

    same_label = 0
    valid_edges = 0
    for u, v in graph.edges():
        if u in labels and v in labels:
            valid_edges += 1
            if labels[u] == labels[v]:
                same_label += 1
    if valid_edges == 0:
        return 0.0
    return same_label / valid_edges


def homophily_gap(
    client_graphs: List[nx.Graph], client_labels: List[Dict[int, int]]
) -> Dict[str, object]:
    """Compute pairwise homophily gaps and federation-level variance."""
    if len(client_graphs) != len(client_labels):
        raise ValueError("client_graphs and client_labels must have equal length")

    homophily_values = [compute_homophily(g, lbl) for g, lbl in zip(client_graphs, client_labels)]
    pairwise_hg = np.abs(np.subtract.outer(homophily_values, homophily_values))
    return {
        "pairwise_hg": pairwise_hg,
        "homophily_per_client": homophily_values,
        "variance": float(np.var(homophily_values)),
    }


def _shared_degree_histograms(client_graphs: List[nx.Graph], bins: Optional[int] = None) -> List[np.ndarray]:
    max_degree = max((max((d for _, d in g.degree()), default=0) for g in client_graphs), default=0)
    if bins is None:
        bins = max(5, min(50, max_degree + 1))
    bin_edges = np.linspace(0, max_degree + 1, bins + 1)

    summaries: List[np.ndarray] = []
    for g in client_graphs:
        degrees = [d for _, d in g.degree()]
        hist, _ = np.histogram(degrees, bins=bin_edges, density=False)
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-12
        summaries.append(hist)
    return summaries


def _shared_clustering_histograms(client_graphs: List[nx.Graph], bins: int = 10) -> List[np.ndarray]:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    summaries: List[np.ndarray] = []
    for g in client_graphs:
        values = list(nx.clustering(g).values()) if g.number_of_nodes() > 0 else [0.0]
        hist, _ = np.histogram(values, bins=bin_edges, density=False)
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-12
        summaries.append(hist)
    return summaries


def _shared_spectral_histograms(client_graphs: List[nx.Graph], bins: int = 20) -> List[np.ndarray]:
    bin_edges = np.linspace(0.0, 2.0, bins + 1)
    summaries: List[np.ndarray] = []
    for g in client_graphs:
        if g.number_of_nodes() == 0:
            hist = np.ones(bins, dtype=float) / bins
        else:
            L = nx.normalized_laplacian_matrix(g).astype(float).toarray()
            evals = np.linalg.eigvalsh(L)
            hist, _ = np.histogram(evals, bins=bin_edges, density=False)
            hist = hist.astype(float)
            hist /= hist.sum() + 1e-12
        summaries.append(hist)
    return summaries


def topological_divergence(
    client_graphs: List[nx.Graph], bins: Optional[int] = None, metric: str = "degree"
) -> Dict[str, object]:
    """Compute pairwise structural divergence using common summary histograms.

    Supported structural summaries:
    - degree histogram
    - local clustering coefficient histogram
    - normalized Laplacian spectral-density histogram
    """
    if not client_graphs:
        raise ValueError("client_graphs must contain at least one graph")

    if metric == "degree":
        summaries = _shared_degree_histograms(client_graphs, bins=bins)
    elif metric == "clustering":
        summaries = _shared_clustering_histograms(client_graphs, bins=bins or 10)
    elif metric == "spectral":
        summaries = _shared_spectral_histograms(client_graphs, bins=bins or 20)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    K = len(client_graphs)
    pairwise_td = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            td = jensen_shannon_divergence(summaries[i], summaries[j])
            pairwise_td[i, j] = pairwise_td[j, i] = td

    return {
        "pairwise_td": pairwise_td,
        "summaries": summaries,
        "metric": metric,
    }
