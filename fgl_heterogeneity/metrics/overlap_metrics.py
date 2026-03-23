import numpy as np
from typing import List, Set, Dict

def neighbor_overlap_index(client_node_sets: List[Set[int]]) -> Dict:
    """
    Compute pairwise NOI (Node Overlap Index).
    """
    K = len(client_node_sets)
    pairwise_noi = np.zeros((K, K))
    
    for i in range(K):
        for j in range(i+1, K):
            intersection = len(client_node_sets[i] & client_node_sets[j])
            union = len(client_node_sets[i] | client_node_sets[j])
            if union > 0:
                pairwise_noi[i, j] = pairwise_noi[j, i] = intersection / union
    
    return {
        'pairwise_noi': pairwise_noi,
        'mean_overlap': np.mean(pairwise_noi[pairwise_noi > 0]) if np.any(pairwise_noi > 0) else 0.0
    }


def missing_neighbor_ratio(client_node_sets: List[Set[int]],
                           global_graph_adj: Dict[int, Set[int]],
                           eps: float = 1e-10) -> List[float]:
    """
    Compute MNR for each client.
    
    Args:
        client_node_sets: List of node sets per client
        global_graph_adj: Adjacency dict mapping node -> set of neighbors in global graph
    """
    mnr_list = []
    for nodes in client_node_sets:
        total_deg = 0
        missing_deg = 0
        for node in nodes:
            if node in global_graph_adj:
                neighbors = global_graph_adj[node]
                total_deg += len(neighbors)
                missing_deg += len(neighbors - nodes)
        mnr = missing_deg / (total_deg + eps)
        mnr_list.append(mnr)
    return mnr_list
