"""Input/output utilities for loading datasets and saving benchmark artifacts."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from .manifest import _to_jsonable

def load_torch_geometric_dataset(name: str, root: str = "./data"):
    """Load a graph dataset from torch_geometric if available."""
    try:
        from torch_geometric.datasets import Planetoid, TUDataset
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("torch and torch_geometric are required for this function") from exc

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=root, name=name)
        return dataset[0]
    return TUDataset(root=root, name=name)


def save_split_manifest(manifest: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(manifest), f, indent=2, sort_keys=True)


def load_split_manifest(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_partition(client_node_sets: List[List[int]], filepath: str) -> None:
    data = {"client_node_sets": [list(s) for s in client_node_sets]}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_partition(filepath: str) -> List[set]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [set(s) for s in data["client_node_sets"]]


def save_embeddings(embeddings: List[np.ndarray], filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings(filepath: str) -> List[np.ndarray]:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
