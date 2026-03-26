"""Boundary-edge handling policies for subgraph federations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Set, Tuple

import networkx as nx


class BoundaryPolicy(Enum):
    INDUCED_SUBGRAPH = "induced_subgraph"
    MASKED_BOUNDARY = "masked_boundary"
    NEIGHBOR_AUGMENTED = "neighbor_augmented"


def apply_boundary_policy(
    global_graph: nx.Graph,
    client_node_sets: List[Set[int]],
    policy: BoundaryPolicy,
    neighbor_augmentation_fn: Any = None,
) -> Tuple[List[nx.Graph], Dict[str, Any]]:
    """Apply a boundary policy and return client-local graph views plus metadata."""
    client_subgraphs: List[nx.Graph] = []
    metadata: Dict[str, Any] = {"policy": policy.value}

    if policy == BoundaryPolicy.INDUCED_SUBGRAPH:
        for nodes in client_node_sets:
            client_subgraphs.append(global_graph.subgraph(nodes).copy())
        metadata["boundary_counts"] = [
            sum(1 for u in nodes for v in global_graph.neighbors(u) if v not in nodes)
            for nodes in client_node_sets
        ]

    elif policy == BoundaryPolicy.MASKED_BOUNDARY:
        boundary_counts = []
        boundary_node_counts = []
        for nodes in client_node_sets:
            subgraph = global_graph.subgraph(nodes).copy()
            count = 0
            masked_nodes = 0
            for u in nodes:
                external_neighbors = [v for v in global_graph.neighbors(u) if v not in nodes]
                if external_neighbors:
                    masked_nodes += 1
                count += len(external_neighbors)
                subgraph.nodes[u]["masked_boundary_degree"] = len(external_neighbors)
            boundary_counts.append(count)
            boundary_node_counts.append(masked_nodes)
            client_subgraphs.append(subgraph)
        metadata["boundary_counts"] = boundary_counts
        metadata["boundary_node_counts"] = boundary_node_counts

    elif policy == BoundaryPolicy.NEIGHBOR_AUGMENTED:
        if neighbor_augmentation_fn is not None:
            client_subgraphs = neighbor_augmentation_fn(global_graph, client_node_sets)
            metadata["augmentation"] = getattr(neighbor_augmentation_fn, "__name__", "custom")
        else:
            synthetic_neighbor_counts = []
            for nodes in client_node_sets:
                subgraph = global_graph.subgraph(nodes).copy()
                synth_count = 0
                for u in nodes:
                    missing = [v for v in global_graph.neighbors(u) if v not in nodes]
                    if missing:
                        synth_count += len(missing)
                        subgraph.nodes[u]["synthetic_neighbor_count"] = len(missing)
                synthetic_neighbor_counts.append(synth_count)
                client_subgraphs.append(subgraph)
            metadata["augmentation"] = "metadata_only_placeholder"
            metadata["synthetic_neighbor_counts"] = synthetic_neighbor_counts

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return client_subgraphs, metadata
