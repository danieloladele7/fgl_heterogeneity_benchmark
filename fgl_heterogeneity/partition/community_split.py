"""Deterministic community-based partitioning for structural skew."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np


def community_split(
    graph: nx.Graph,
    num_clients: int,
    resolution: float = 1.0,
    seed: int = 42,
    algorithm: str = "louvain",
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """Split a graph into communities and greedily assign them to clients.

    The protocol is deterministic under a fixed seed, NetworkX version, and graph.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    if algorithm != "louvain":
        raise ValueError("Only 'louvain' is supported in this implementation")

    communities = list(
        nx.community.louvain_communities(graph, resolution=resolution, seed=seed)
    )
    communities = [set(comm) for comm in communities]
    communities.sort(key=len, reverse=True)

    client_node_sets: List[Set[int]] = [set() for _ in range(num_clients)]
    client_sizes = [0 for _ in range(num_clients)]
    community_sizes = []

    for idx, comm in enumerate(communities):
        min_client = int(np.argmin(client_sizes))
        client_node_sets[min_client].update(comm)
        client_sizes[min_client] += len(comm)
        community_sizes.append({"community_id": idx, "size": len(comm)})

    metadata = {
        "protocol": "community_split",
        "algorithm": algorithm,
        "community_library": f"networkx_{nx.__version__}",
        "num_clients": num_clients,
        "resolution": resolution,
        "seed": seed,
        "num_communities": len(communities),
        "per_client_nodes": [len(c) for c in client_node_sets],
        "community_sizes": community_sizes,
    }
    return client_node_sets, metadata
