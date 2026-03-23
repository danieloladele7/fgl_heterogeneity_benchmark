import numpy as np
from scipy.special import rel_entr
from typing import List, Dict, Tuple

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    Args:
        p, q: Probability vectors (normalized counts)
        eps: Small constant for numerical stability
        
    Returns:
        JSD value in [0, log(2)]
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (rel_entr(p, m).sum() + rel_entr(q, m).sum())


def label_distribution_divergence(client_label_counts: List[np.ndarray]) -> Dict:
    """
    Compute pairwise JSD and global LPV for label distributions.
    
    Args:
        client_label_counts: List of per-client class count vectors (C-dimensional)
        
    Returns:
        Dict with 'pairwise_jsd' (KxK matrix), 'lpv' (scalar), 'jsd_per_client' (list)
    """
    K = len(client_label_counts)
    C = client_label_counts[0].shape[0]
    
    # Normalize to probability distributions
    probs = [counts / counts.sum() for counts in client_label_counts]
    
    # Pairwise JSD
    pairwise_jsd = np.zeros((K, K))
    for i in range(K):
        for j in range(i+1, K):
            jsd = jensen_shannon_divergence(probs[i], probs[j])
            pairwise_jsd[i, j] = pairwise_jsd[j, i] = jsd
    
    # Label proportion variance (LPV)
    label_proportion_matrix = np.array([p for p in probs])  # K x C
    lpv = np.mean(np.var(label_proportion_matrix, axis=0))
    
    # Per-client average JSD
    jsd_per_client = [np.mean(pairwise_jsd[i, [j for j in range(K) if j != i]]) for i in range(K)]
    
    return {
        'pairwise_jsd': pairwise_jsd,
        'lpv': lpv,
        'jsd_per_client': jsd_per_client
    }
