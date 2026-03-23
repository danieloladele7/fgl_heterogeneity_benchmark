import numpy as np
from sklearn.decomposition import PCA
from typing import List, Optional

def embedding_centroid_divergence(embeddings_list: List[np.ndarray], eps: float = 1e-8) -> Dict:
    """
    Compute ECD (Embedding Centroid Divergence) between clients.
    
    Args:
        embeddings_list: List of [n_k x d] embedding matrices per client
        eps: Small constant for stability
        
    Returns:
        Dict with pairwise ECD matrix and per-client stats
    """
    K = len(embeddings_list)
    d = embeddings_list[0].shape[1]
    
    centroids = []
    radii = []
    for emb in embeddings_list:
        mu = np.mean(emb, axis=0)
        tau = np.mean(np.sum((emb - mu) ** 2, axis=1))
        centroids.append(mu)
        radii.append(tau)
    
    pairwise_ecd = np.zeros((K, K))
    for i in range(K):
        for j in range(i+1, K):
            denom = np.sqrt(radii[i] + radii[j] + eps)
            ecd = np.linalg.norm(centroids[i] - centroids[j]) / denom
            pairwise_ecd[i, j] = pairwise_ecd[j, i] = ecd
    
    return {
        'pairwise_ecd': pairwise_ecd,
        'centroids': centroids,
        'radii': radii
    }


def principal_subspace_divergence(embeddings_list: List[np.ndarray], rank: int = 10) -> Dict:
    """
    Compute PSD (Principal Subspace Divergence) between clients.
    
    Args:
        embeddings_list: List of [n_k x d] embedding matrices per client
        rank: Number of principal components to retain
        
    Returns:
        Dict with pairwise PSD matrix
    """
    K = len(embeddings_list)
    d = embeddings_list[0].shape[1]
    r = min(rank, d)
    
    pairwise_psd = np.zeros((K, K))
    
    for i in range(K):
        # Center embeddings
        emb_i = embeddings_list[i] - np.mean(embeddings_list[i], axis=0)
        # PCA to get top-r orthonormal basis
        U_i, _, _ = np.linalg.svd(emb_i, full_matrices=False)
        U_i = U_i[:, :r]  # n_i x r
        Pi = U_i @ U_i.T  # projection matrix (n_i x n_i) - but note: different dimensions across clients?
        # Actually we need per-client projection matrices; but for comparability we compute on the same dimension
        # Standard approach: compute subspace basis U_i (d x r) then compare via principal angles
        # For different n_k, we compare subspaces in the embedding dimension d.
        
        # Alternative: use the d x r basis and compute projection distance
        # We'll store the d x r basis instead
        pass
    
    # For simplicity, we implement using the d x r basis approach
    # Each client: basis matrix U_k (d x r)
    bases = []
    for emb in embeddings_list:
        emb_centered = emb - np.mean(emb, axis=0)
        # Compute SVD; keep top-r right singular vectors (d x r)
        # Actually SVD: emb = U @ S @ V^T; V is d x d
        _, _, Vt = np.linalg.svd(emb_centered, full_matrices=False)
        bases.append(Vt[:r, :].T)  # d x r
    
    for i in range(K):
        for j in range(i+1, K):
            # Compute Frobenius norm of projection difference
            Pi = bases[i] @ bases[i].T
            Pj = bases[j] @ bases[j].T
            diff_norm = np.linalg.norm(Pi - Pj, ord='fro')
            pairwise_psd[i, j] = pairwise_psd[j, i] = diff_norm / np.sqrt(2 * r)
    
    return {
        'pairwise_psd': pairwise_psd
    }
