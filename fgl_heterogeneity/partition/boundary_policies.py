from enum import Enum
from typing import List, Set, Dict, Tuple, Any

class BoundaryPolicy(Enum):
    """Policy for handling cross-client edges."""
    INDUCED_SUBGRAPH = "induced_subgraph"
    MASKED_BOUNDARY = "masked_boundary"
    NEIGHBOR_AUGMENTED = "neighbor_augmented"

def apply_boundary_policy(
    global_graph: Any,
    client_node_sets: List[Set[int]],
    policy: BoundaryPolicy,
    neighbor_augmentation_fn: Any = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Apply boundary-edge policy to create client subgraphs.
    
    Args:
        global_graph: NetworkX graph
        client_node_sets: Node sets per client
        policy: BoundaryPolicy enum
        neighbor_augmentation_fn: Function that generates augmented neighbors
    
    Returns:
        client_subgraphs: List of client subgraphs (NetworkX graphs)
        metadata: Policy details
    """
    client_subgraphs = []
    
    if policy == BoundaryPolicy.INDUCED_SUBGRAPH:
        for nodes in client_node_sets:
            subgraph = global_graph.subgraph(nodes)
            client_subgraphs.append(subgraph)
    elif policy == BoundaryPolicy.MASKED_BOUNDARY:
        # Return subgraph but also compute boundary edge counts (no neighbor identities)
        boundary_counts = []
        for nodes in client_node_sets:
            subgraph = global_graph.subgraph(nodes)
            # Count boundary edges
            count = 0
            for u in nodes:
                for v in global_graph.neighbors(u):
                    if v not in nodes:
                        count += 1
            boundary_counts.append(count)
            client_subgraphs.append(subgraph)
        # Store boundary_counts in metadata
    elif policy == BoundaryPolicy.NEIGHBOR_AUGMENTED:
        if neighbor_augmentation_fn is None:
            # Default: add dummy nodes to represent missing neighbors
            client_subgraphs = []
            for nodes in client_node_sets:
                subgraph = global_graph.subgraph(nodes).copy()
                # Add placeholder nodes for each missing neighbor
                # (Simplified: just count; in practice, generate synthetic features)
                client_subgraphs.append(subgraph)
        else:
            client_subgraphs = neighbor_augmentation_fn(global_graph, client_node_sets)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    metadata = {
        'policy': policy.value
    }
    
    return client_subgraphs, metadata
