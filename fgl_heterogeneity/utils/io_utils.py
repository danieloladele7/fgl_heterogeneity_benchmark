"""
Input/Output utilities for loading datasets and saving splits.
"""

import json
import os
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Union

# For loading torch geometric datasets
try:
    import torch
    from torch_geometric.datasets import Planetoid, TUDataset
    from torch_geometric.utils import to_networkx
except ImportError:
    torch = None
    to_networkx = None


def load_torch_geometric_dataset(name: str, root: str = './data'):
    """
    Load a graph dataset from torch geometric.

    Args:
        name: Dataset name (e.g., 'Cora', 'CiteSeer', 'PubMed', 'PROTEINS', etc.)
        root: Root directory for dataset.

    Returns:
        data: PyTorch Geometric Data object (single graph for node tasks)
        or list of Data objects (for graph tasks).
    """
    if torch is None:
        raise ImportError("torch and torch_geometric are required for this function.")

    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
        return data
    else:
        # For TUDataset
        dataset = TUDataset(root=root, name=name)
        return dataset  # list of graphs


def save_split_manifest(manifest: Dict[str, Any], filepath: str):
    """Save manifest to JSON."""
    with open(filepath, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_split_manifest(filepath: str) -> Dict[str, Any]:
    """Load manifest from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_partition(client_node_sets: List[List[int]], filepath: str):
    """
    Save client node sets to a JSON file.
    """
    data = {
        'client_node_sets': [list(s) for s in client_node_sets]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)


def load_partition(filepath: str) -> List[set]:
    """
    Load client node sets from JSON.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [set(s) for s in data['client_node_sets']]


def save_embeddings(embeddings: List[np.ndarray], filepath: str):
    """
    Save list of embedding matrices to a pickle file.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)


def load_embeddings(filepath: str) -> List[np.ndarray]:
    """
    Load embeddings from pickle.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_directory(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
