import numpy as np
from typing import Callable, Dict, Any, List

def aggregate_pairwise_metrics(pairwise_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute federation-level summaries from pairwise metrics.
    """
    K = pairwise_matrix.shape[0]
    # Extract upper triangle (excluding diagonal)
    triu_vals = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    mean_val = np.mean(triu_vals)
    max_val = np.max(triu_vals)
    
    # Per-client average distance to others
    per_client = []
    for i in range(K):
        others = [pairwise_matrix[i, j] for j in range(K) if j != i]
        per_client.append(np.mean(others) if others else 0.0)
    
    return {
        'mean_pairwise': mean_val,
        'max_pairwise': max_val,
        'per_client_mean': per_client,
        'variance': np.var(triu_vals) if len(triu_vals) > 0 else 0.0
    }


def compute_full_heterogeneity_profile(
    label_counts: List[np.ndarray],
    embeddings: List[np.ndarray],
    graphs: List[Any],
    node_labels: List[Dict[int, int]],
    node_sets: List[Set[int]],
    global_adj: Dict[int, Set[int]],
    sample_sizes: List[int],
    subspace_rank: int = 10
) -> Dict[str, Any]:
    """
    Compute all metrics and return aggregated profile.
    """
    from .label_metrics import label_distribution_divergence
    from .representation_metrics import embedding_centroid_divergence, principal_subspace_divergence
    from .topology_metrics import homophily_gap, topological_divergence
    from .overlap_metrics import neighbor_overlap_index, missing_neighbor_ratio
    from .quantity_metrics import quantity_imbalance_index
    
    profile = {}
    
    # M1: Label skew
    label_res = label_distribution_divergence(label_counts)
    profile['label_jsd'] = aggregate_pairwise_metrics(label_res['pairwise_jsd'])
    profile['lpv'] = label_res['lpv']
    
    # M2: ECD
    if embeddings is not None:
        ecd_res = embedding_centroid_divergence(embeddings)
        profile['ecd'] = aggregate_pairwise_metrics(ecd_res['pairwise_ecd'])
        
        # M3: PSD
        psd_res = principal_subspace_divergence(embeddings, rank=subspace_rank)
        profile['psd'] = aggregate_pairwise_metrics(psd_res['pairwise_psd'])
    
    # M4: Homophily gap
    if graphs is not None and node_labels is not None:
        hg_res = homophily_gap(graphs, node_labels)
        profile['homophily_gap'] = aggregate_pairwise_metrics(hg_res['pairwise_hg'])
        profile['homophily_variance'] = hg_res['variance']
    
    # M5: Topological divergence (degree histogram default)
    if graphs is not None:
        td_res = topological_divergence(graphs, metric='degree')
        profile['topo_divergence'] = aggregate_pairwise_metrics(td_res['pairwise_td'])
    
    # M6: Overlap metrics
    if node_sets is not None:
        noi_res = neighbor_overlap_index(node_sets)
        profile['noi'] = noi_res['mean_overlap']
        if global_adj is not None:
            mnr_list = missing_neighbor_ratio(node_sets, global_adj)
            profile['mnr_per_client'] = mnr_list
            profile['mnr_mean'] = np.mean(mnr_list) if mnr_list else 0.0
    
    # M7: Quantity skew
    qii_res = quantity_imbalance_index(sample_sizes)
    profile['qii'] = qii_res['qii']
    
    # M8: Domain shift profile (vector summary)
    profile['domain_shift_profile'] = {
        'mean_jsd': profile['label_jsd']['mean_pairwise'],
        'mean_ecd': profile.get('ecd', {}).get('mean_pairwise', 0.0),
        'mean_psd': profile.get('psd', {}).get('mean_pairwise', 0.0),
        'mean_hg': profile.get('homophily_gap', {}).get('mean_pairwise', 0.0),
        'mean_td': profile.get('topo_divergence', {}).get('mean_pairwise', 0.0),
        'mean_mnr': profile.get('mnr_mean', 0.0),
        'qii': profile['qii']
    }
    
    return profile
