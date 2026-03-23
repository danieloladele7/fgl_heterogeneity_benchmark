import numpy as np
import networkx as nx
from typing import List, Set, Tuple, Dict, Any, Optional

def community_split(
    graph: nx.Graph,
    num_clients: int,
    resolution: float = 1.0,
    seed: int = 42,
    algorithm: str = 'louvain'
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Split graph into communities and assign to clients (balanced assignment).
    
    Args:
        graph: Global graph
        num_clients: Number of clients
        resolution: Resolution parameter for community detection
        seed: Random seed
        algorithm: 'louvain' or 'leiden'
    
    Returns:
        client_node_sets: List of node sets per client
        metadata: Split parameters and community stats
    """
    np.random.seed(seed)
    
    # Detect communities
    if algorithm == 'louvain':
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(graph, resolution=resolution, random_state=seed)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, set()).add(node)
        community_list = list(communities.values())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Sort communities by size descending
    community_list.sort(key=len, reverse=True)
    
    # Assign to clients greedily (smallest client first)
    client_node_sets = [set() for _ in range(num_clients)]
    for comm in community_list:
        # Find client with smallest current size
        client_sizes = [len(c) for c in client_node_sets]
        min_client = np.argmin(client_sizes)
        client_node_sets[min_client].update(comm)
    
    metadata = {
        'protocol': 'community_split',
        'algorithm': algorithm,
        'num_clients': num_clients,
        'resolution': resolution,
        'seed': seed,
        'num_communities': len(community_list),
        'per_client_nodes': [len(c) for c in client_node_sets]
    }
    
    return client_node_sets, metadata
