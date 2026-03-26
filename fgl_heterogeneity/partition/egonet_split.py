"""Ego-net-based partitioning for overlap and missing-neighbor heterogeneity."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np


def _khop_neighbors(graph: nx.Graph, anchor: int, hop_radius: int) -> Set[int]:
    visited: Set[int] = set()
    queue = deque([(anchor, 0)])
    while queue:
        node, dist = queue.popleft()
        if node in visited or dist > hop_radius:
            continue
        visited.add(node)
        if dist == hop_radius:
            continue
        for nb in graph.neighbors(node):
            if nb not in visited:
                queue.append((nb, dist + 1))
    return visited


def ego_net_split(
    graph: nx.Graph,
    num_clients: int,
    anchors_per_client: int,
    hop_radius: int,
    seed: int = 42,
    overlap_strategy: str = "partial",
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """Build client subgraphs from unions of anchor ego-nets.

    overlap_strategy controls anchor selection, not guaranteed node-set disjointness:
    - 'disjoint': disjoint anchors sampled without replacement globally
    - 'partial': anchors sampled without replacement per client
    - 'full': anchors sampled with replacement per client
    """
    if num_clients <= 0 or anchors_per_client <= 0:
        raise ValueError("num_clients and anchors_per_client must be positive")

    rng = np.random.default_rng(seed)
    nodes = np.asarray(list(graph.nodes()))
    total_anchors = num_clients * anchors_per_client
    if overlap_strategy == "disjoint" and total_anchors > len(nodes):
        raise ValueError("Not enough nodes for globally disjoint anchors")

    anchors_per_client_list: List[List[int]] = []
    if overlap_strategy == "disjoint":
        anchors = rng.choice(nodes, size=total_anchors, replace=False)
        anchors_per_client_list = [
            list(map(int, anchors[i * anchors_per_client : (i + 1) * anchors_per_client]))
            for i in range(num_clients)
        ]
    elif overlap_strategy == "partial":
        for _ in range(num_clients):
            anchors = rng.choice(nodes, size=anchors_per_client, replace=False)
            anchors_per_client_list.append(list(map(int, anchors)))
    elif overlap_strategy == "full":
        for _ in range(num_clients):
            anchors = rng.choice(nodes, size=anchors_per_client, replace=True)
            anchors_per_client_list.append(list(map(int, anchors)))
    else:
        raise ValueError("overlap_strategy must be 'disjoint', 'partial', or 'full'")

    client_node_sets: List[Set[int]] = []
    for anchors in anchors_per_client_list:
        nodes_set: Set[int] = set()
        for a in anchors:
            nodes_set.update(_khop_neighbors(graph, a, hop_radius))
        client_node_sets.append(nodes_set)

    metadata = {
        "protocol": "egonet_split",
        "num_clients": num_clients,
        "anchors_per_client": anchors_per_client,
        "hop_radius": hop_radius,
        "seed": seed,
        "overlap_strategy": overlap_strategy,
        "anchor_selection_semantics": "disjoint_anchors_not_necessarily_disjoint_egonets"
        if overlap_strategy == "disjoint"
        else "clientwise_anchor_sampling",
        "anchors": anchors_per_client_list,
        "per_client_nodes": [len(c) for c in client_node_sets],
    }
    return client_node_sets, metadata
