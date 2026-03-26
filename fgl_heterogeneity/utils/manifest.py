"""Manifest generation utilities for reproducible split release."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Set

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def generate_manifest(
    client_node_sets: Optional[List[Set[int]]] = None,
    client_graph_indices: Optional[List[List[int]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a manifest and append a deterministic SHA-256 hash."""
    manifest: Dict[str, Any] = {}

    if client_node_sets is not None:
        manifest["client_nodes"] = [sorted(list(s)) for s in client_node_sets]
    if client_graph_indices is not None:
        manifest["client_graph_indices"] = [list(idx) for idx in client_graph_indices]
    if metadata is not None:
        manifest["metadata"] = _to_jsonable(metadata)

    manifest_str = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["hash"] = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
    return manifest


def save_manifest(manifest: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(manifest), f, indent=2, sort_keys=True)


def load_manifest(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
