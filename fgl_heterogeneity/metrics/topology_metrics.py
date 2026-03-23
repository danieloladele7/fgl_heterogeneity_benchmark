import numpy as np
import networkx as nx
from scipy.special import rel_entr
from typing import List, Dict, Union, Optional

def compute_homophily(graph: nx.Graph, labels: Dict[int, int]) -> float:
    """
    Compute edge homophily ratio for a graph.
    
    Args:
        graph: NetworkX graph
        labels: Dictionary mapping node ID to class label
        
    Returns:
        Homophily ratio = (# same-label edges) / (total edges)
    """
    if graph.number_of_edges() == 0:
        return 0.0
    
    same_label = 0
    for u, v in graph.edges():
        if u in labels and v in labels and labels[u] == labels[v]:
            same_label += 1
    return same_label / graph.number_of_edges()


def homophily_gap(client_graphs: List[nx.Graph], 
                  client_labels: List[Dict[int, int]]) -> Dict:
    """
    Compute pairwise homophily gap and federation-level variance.
    """
    K = len(client_graphs)
    homophily_values = []
    for g, lbl in zip(client_graphs, client_labels):
        homophily_values.append(compute_homophily(g, lbl))
    
    pairwise_hg = np.abs(np.subtract.outer(homophily_values, homophily_values))
    var_hg = np.var(homophily_values)
    
    return {
        'pairwise_hg': pairwise_hg,
        'homophily_per_client': homophily_values,
        'variance': var_hg
    }


def degree_histogram(graph: nx.Graph, bins: Optional[int] = None) -> np.ndarray:
    """
    Compute normalized degree histogram.
    """
    degrees = [d for _, d in graph.degree()]
    if bins is None:
        bins = int(np.sqrt(len(degrees)))  # heuristic
    hist, _ = np.histogram(degrees, bins=bins, density=True)
    return hist / (hist.sum() + 1e-10)


def topological_divergence(client_graphs: List[nx.Graph], 
                           bins: Optional[int] = None,
                           metric: str = 'degree') -> Dict:
    """
    Compute pairwise topological divergence via JSD on structural summaries.
    
    Args:
        client_graphs: List of NetworkX graphs
        bins: Number of bins for histogram
        metric: 'degree', 'spectral', or 'motif'
        
    Returns:
        Dict with pairwise TD matrix
    """
    K = len(client_graphs)
    summaries = []
    
    if metric == 'degree':
        for g in client_graphs:
            summaries.append(degree_histogram(g, bins))
    elif metric == 'spectral':
        # Simplified: use normalized Laplacian eigenvalue histogram
        for g in client_graphs:
            try:
                L = nx.normalized_laplacian_matrix(g).todense()
                evals = np.linalg.eigvalsh(L)
                hist, _ = np.histogram(evals, bins=20, density=True)
                summaries.append(hist / (hist.sum() + 1e-10))
            except:
                summaries.append(np.ones(20) / 20)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Compute pairwise JSD
    pairwise_td = np.zeros((K, K))
    for i in range(K):
        for j in range(i+1, K):
            p = summaries[i] + 1e-10
            q = summaries[j] + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            jsd = 0.5 * (rel_entr(p, m).sum() + rel_entr(q, m).sum())
            pairwise_td[i, j] = pairwise_td[j, i] = jsd
    
    return {
        'pairwise_td': pairwise_td,
        'summaries': summaries
    }
