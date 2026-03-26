"""Representation-based heterogeneity metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _validate_embeddings(embeddings_list: List[np.ndarray]) -> None:
    if not embeddings_list:
        raise ValueError("embeddings_list must contain at least one client matrix")
    dims = {emb.shape[1] for emb in embeddings_list}
    if len(dims) != 1:
        raise ValueError("All embedding matrices must share the same feature dimension")


def embedding_centroid_divergence(
    embeddings_list: List[np.ndarray], eps: float = 1e-8
) -> Dict[str, object]:
    """Compute the embedding centroid divergence (ECD).

    For each client k with embedding matrix Z_k \in R^{n_k x d}, this computes
    the centroid mu_k and average squared radius tau_k, then returns the pairwise
    divergence

        ECD(k,k') = ||mu_k - mu_k'||_2 / sqrt(tau_k + tau_k' + eps).
    """
    _validate_embeddings(embeddings_list)

    K = len(embeddings_list)
    centroids: List[np.ndarray] = []
    radii: List[float] = []

    for emb in embeddings_list:
        if emb.ndim != 2:
            raise ValueError("Each embedding matrix must be 2-dimensional")
        mu = np.mean(emb, axis=0)
        tau = float(np.mean(np.sum((emb - mu) ** 2, axis=1)))
        centroids.append(mu)
        radii.append(tau)

    pairwise_ecd = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            denom = np.sqrt(radii[i] + radii[j] + eps)
            ecd = float(np.linalg.norm(centroids[i] - centroids[j]) / denom)
            pairwise_ecd[i, j] = pairwise_ecd[j, i] = ecd

    return {
        "pairwise_ecd": pairwise_ecd,
        "centroids": centroids,
        "radii": radii,
    }


def principal_subspace_divergence(
    embeddings_list: List[np.ndarray], rank: int = 10
) -> Dict[str, object]:
    """Compute the principal subspace divergence (PSD).

    For each client k, compute the top-r right singular vectors of the centered
    embedding matrix Z_k, form the projection matrix Pi_k = U_k U_k^T in the
    shared embedding space, and compare clients via

        PSD(k,k') = ||Pi_k - Pi_k'||_F / sqrt(2r).
    """
    _validate_embeddings(embeddings_list)

    K = len(embeddings_list)
    d = embeddings_list[0].shape[1]
    r = max(1, min(rank, d))

    bases: List[np.ndarray] = []
    effective_ranks: List[int] = []

    for emb in embeddings_list:
        if emb.ndim != 2:
            raise ValueError("Each embedding matrix must be 2-dimensional")
        emb_centered = emb - np.mean(emb, axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(emb_centered, full_matrices=False)
        available_rank = int(min(r, np.sum(s > 1e-12), vt.shape[0]))
        if available_rank == 0:
            basis = np.zeros((d, 1), dtype=float)
            effective_ranks.append(1)
        else:
            basis = vt[:available_rank, :].T
            effective_ranks.append(available_rank)
        bases.append(basis)

    pairwise_psd = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            ri = effective_ranks[i]
            rj = effective_ranks[j]
            denom_rank = max(1, min(ri, rj))
            Pi = bases[i] @ bases[i].T
            Pj = bases[j] @ bases[j].T
            diff_norm = np.linalg.norm(Pi - Pj, ord="fro")
            pairwise_psd[i, j] = pairwise_psd[j, i] = float(
                diff_norm / np.sqrt(2.0 * denom_rank)
            )

    return {
        "pairwise_psd": pairwise_psd,
        "bases": bases,
        "effective_ranks": effective_ranks,
    }
