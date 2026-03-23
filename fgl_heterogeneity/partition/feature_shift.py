import numpy as np
from typing import List, Dict, Any, Callable, Tuple

def apply_feature_shift(
    feature_matrices: List[np.ndarray],
    client_assignments: List[Any],
    shift_type: str,
    shift_intensity: float,
    seed: int
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Apply client-specific feature transformations.
    
    Args:
        feature_matrices: List of [n_k x d] feature matrices per client
        client_assignments: Original client IDs or grouping
        shift_type: 'affine', 'gaussian_noise', 'masking', 'sign_perturb'
        shift_intensity: Strength of transformation
        seed: Random seed
    
    Returns:
        transformed_features: List of transformed matrices
        metadata: Transformation parameters
    """
    np.random.seed(seed)
    transformed = []
    
    for feats in feature_matrices:
        if shift_type == 'affine':
            # Scale + shift
            scale = 1 + shift_intensity * np.random.randn()
            shift = shift_intensity * np.random.randn(feats.shape[1])
            transformed_feats = feats * scale + shift
        elif shift_type == 'gaussian_noise':
            noise = shift_intensity * np.random.randn(*feats.shape)
            transformed_feats = feats + noise
        elif shift_type == 'masking':
            mask = np.random.rand(*feats.shape) > shift_intensity
            transformed_feats = feats * mask
        elif shift_type == 'sign_perturb':
            # Randomly flip sign with probability shift_intensity/2
            flip_mask = np.random.rand(*feats.shape) < (shift_intensity / 2)
            transformed_feats = np.where(flip_mask, -feats, feats)
        else:
            raise ValueError(f"Unknown shift_type: {shift_type}")
        
        transformed.append(transformed_feats)
    
    metadata = {
        'protocol': 'feature_shift',
        'shift_type': shift_type,
        'shift_intensity': shift_intensity,
        'seed': seed
    }
    
    return transformed, metadata
