"""Cross-domain federation protocols."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _as_domain_payload(dataset: Any) -> Dict[str, Any]:
    if isinstance(dataset, dict) and "domain_id" in dataset:
        return dataset
    return {"domain_id": None, "payload": dataset}


def cross_domain_federation(
    datasets: List[Any],
    num_clients: int,
    assignment: str = "one_per_client",
    seed: int = 42,
    domain_ids: Optional[List[List[int]]] = None,
    mixing_ratios: Optional[List[List[float]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create a deterministic cross-domain federation.

    Returns a list of per-client dictionaries with explicit domain identifiers.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not datasets:
        raise ValueError("datasets must contain at least one domain")

    rng = np.random.default_rng(seed)
    num_domains = len(datasets)
    normalized_domains = []
    for idx, ds in enumerate(datasets):
        payload = _as_domain_payload(ds)
        payload["domain_id"] = idx if payload["domain_id"] is None else payload["domain_id"]
        normalized_domains.append(payload)

    client_datasets: List[Dict[str, Any]] = []

    if assignment == "one_per_client":
        domain_order = [i % num_domains for i in range(num_clients)]
        for client_id, domain_idx in enumerate(domain_order):
            client_datasets.append(
                {
                    "client_id": client_id,
                    "domain_ids": [normalized_domains[domain_idx]["domain_id"]],
                    "payloads": [normalized_domains[domain_idx]["payload"]],
                    "assignment": "one_per_client",
                }
            )

    elif assignment == "one_to_many":
        # Deterministic round-robin coverage of all domains.
        domain_buckets: List[List[int]] = [[] for _ in range(num_clients)]
        permuted = list(range(num_domains))
        if num_domains > 1:
            permuted = list(rng.permutation(permuted))
        for idx, domain_idx in enumerate(permuted):
            domain_buckets[idx % num_clients].append(domain_idx)
        # if more clients than domains, cycle so no client is empty
        for client_id in range(num_clients):
            if not domain_buckets[client_id]:
                domain_buckets[client_id].append(permuted[client_id % num_domains])
            client_datasets.append(
                {
                    "client_id": client_id,
                    "domain_ids": [normalized_domains[d]["domain_id"] for d in domain_buckets[client_id]],
                    "payloads": [normalized_domains[d]["payload"] for d in domain_buckets[client_id]],
                    "assignment": "one_to_many",
                }
            )

    elif assignment == "mixed":
        if domain_ids is None or mixing_ratios is None:
            raise ValueError("For 'mixed', domain_ids and mixing_ratios must be provided")
        if len(domain_ids) != num_clients or len(mixing_ratios) != num_clients:
            raise ValueError("domain_ids and mixing_ratios must be specified per client")

        for client_id in range(num_clients):
            if len(domain_ids[client_id]) != len(mixing_ratios[client_id]):
                raise ValueError("Each client's domain_ids and mixing_ratios must align")
            ratios = np.asarray(mixing_ratios[client_id], dtype=float)
            if np.any(ratios < 0):
                raise ValueError("mixing ratios must be nonnegative")
            ratios = ratios / ratios.sum()
            client_datasets.append(
                {
                    "client_id": client_id,
                    "domain_ids": [normalized_domains[d]["domain_id"] for d in domain_ids[client_id]],
                    "payloads": [normalized_domains[d]["payload"] for d in domain_ids[client_id]],
                    "mixing_ratios": ratios.tolist(),
                    "assignment": "mixed",
                }
            )
    else:
        raise ValueError(f"Unknown assignment: {assignment}")

    metadata = {
        "protocol": "crossdomain_split",
        "assignment": assignment,
        "num_clients": num_clients,
        "num_domains": num_domains,
        "seed": seed,
        "client_domain_map": [client["domain_ids"] for client in client_datasets],
    }
    if mixing_ratios is not None:
        metadata["mixing_ratios"] = mixing_ratios
    if domain_ids is not None:
        metadata["domain_ids"] = domain_ids

    return client_datasets, metadata
