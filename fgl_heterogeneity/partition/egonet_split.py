import numpy as np
import networkx as nx
from typing import List, Set, Tuple, Dict, Any

def ego_net_split(
    graph: nx.Graph,
    num_clients: int,
    anchors_per_client: int,
    hop_radius: int,
    seed: int,
    overlap_strategy: str = 'disjoint'  # 'disjoint', 'partial', 'full'
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Create overlapping subgraphs via ego-net extraction.
    
    Args:
        graph: Global graph
        num_clients: Number of clients
        anchors_per_client: Number of anchor nodes per client
        hop_radius: Number of hops to include
        seed: Random seed
        overlap_strategy: How to select anchors
    
    Returns:
        client_node_sets: List of node sets per client
    """
    np.random.seed(seed)
    nodes = list(graph.nodes())
    total_anchors = num_clients * anchors_per_client
    
    if total_anchors > len(nodes) and overlap_strategy != 'full':
        raise ValueError("Not enough nodes; use overlap_strategy='full'")
    
    # Select anchors
    if overlap_strategy == 'disjoint':
        # Partition nodes and assign disjoint anchor sets
        shuffled_nodes = np.random.permutation(nodes)
        anchor_groups = np.array_split(shuffled_nodes, num_clients)
        anchors_per_client_list = [list(group[:anchors_per_client]) for group in anchor_groups]
    elif overlap_strategy == 'partial':
        # Sample with replacement or allow overlap
        anchors_per_client_list = []
        for _ in range(num_clients):
            anchors = list(np.random.choice(nodes, size=anchors_per_client, replace=False))
            anchors_per_client_list.append(anchors)
    else:  # full overlap
        anchors_per_client_list = [list(np.random.choice(nodes, size=anchors_per_client, replace=False)) 
                                   for _ in range(num_clients)]
    
    # Build ego-networks
    client_node_sets = []
    for anchors in anchors_per_client_list:
        nodes_set = set()
        for a in anchors:
            # BFS up to hop_radius
            visited = set()
            queue = [(a, 0)]
            while queue:
                node, dist = queue.pop(0)
                if node in visited or dist > hop_radius:
                    continue
                visited.add(node)
                for nb in graph.neighbors(node):
                    if nb not in visited:
                        queue.append((nb, dist + 1))
            nodes_set.update(visited)
        client_node_sets.append(nodes_set)
    
    metadata = {
        'protocol': 'egonet_split',
        'num_clients': num_clients,
        'anchors_per_client': anchors_per_client,
        'hop_radius': hop_radius,
        'seed': seed,
        'overlap_strategy': overlap_strategy,
        'per_client_nodes': [len(c) for c in client_node_sets]
    }
    
    return client_node_sets, metadata
