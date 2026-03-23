import numpy as np
from typing import List, Dict

def quantity_imbalance_index(client_sample_sizes: List[int]) -> Dict:
    """
    Compute QII (Coefficient of Variation of data volumes).
    """
    n = np.array(client_sample_sizes)
    mean_n = np.mean(n)
    if mean_n == 0:
        return {'qii': 0.0, 'mean': 0.0, 'std': 0.0}
    std_n = np.std(n)
    qii = std_n / mean_n
    return {
        'qii': qii,
        'mean': mean_n,
        'std': std_n,
        'per_client': client_sample_sizes
    }
