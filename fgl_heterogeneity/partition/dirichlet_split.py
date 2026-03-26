"""Dirichlet-based label allocation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def dirichlet_label_split(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    max_retries: int = 100,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Partition labeled instances using class-conditional Dirichlet allocation.

    Guarantees that every instance is assigned exactly once and class totals are preserved.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")

    unique_classes = np.unique(labels)
    for _ in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]
        per_class_allocations: Dict[int, List[int]] = {}

        for c in unique_classes:
            class_indices = np.where(labels == c)[0]
            class_indices = rng.permutation(class_indices)

            proportions = rng.dirichlet([alpha] * num_clients)
            cut_points = np.floor(np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            splits = np.split(class_indices, cut_points)
            alloc = []
            for k, chunk in enumerate(splits):
                client_indices[k].extend(chunk.tolist())
                alloc.append(int(len(chunk)))
            per_class_allocations[int(c)] = alloc

        if min(len(idx) for idx in client_indices) >= min_samples_per_client:
            break
    else:
        # Deterministic rebalancing fallback: move the smallest available indices
        # from the largest clients until all clients satisfy the minimum.
        for k in range(num_clients):
            while len(client_indices[k]) < min_samples_per_client:
                donor = int(np.argmax([len(idx) for idx in client_indices]))
                if donor == k or len(client_indices[donor]) <= min_samples_per_client:
                    break
                moved = min(client_indices[donor])
                client_indices[donor].remove(moved)
                client_indices[k].append(moved)

    # Deterministically sort client indices for auditable manifests.
    client_arrays = [np.array(sorted(idx), dtype=int) for idx in client_indices]

    metadata = {
        "protocol": "dirichlet_label_split",
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "min_samples_per_client": min_samples_per_client,
        "max_retries": max_retries,
        "per_client_counts": [int(len(idx)) for idx in client_arrays],
        "per_class_allocations": per_class_allocations,
    }
    return client_arrays, metadata
