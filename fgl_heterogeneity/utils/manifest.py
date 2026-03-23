import hashlib
import json
from typing import Any, Dict, List, Set
import numpy as np

def generate_manifest(
    client_node_sets: List[Set[int]],
    client_graph_indices: List[List[int]] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a reproducible manifest for a split.
    
    Args:
        client_node_sets: For subgraph splits
        client_graph_indices: For graph-level splits
        metadata: Additional parameters
    
    Returns:
        Manifest dictionary with hash
    """
    manifest = {}
    
    if client_node_sets is not None:
        # Convert sets to sorted lists for JSON
        manifest['client_nodes'] = [sorted(list(s)) for s in client_node_sets]
    
    if client_graph_indices is not None:
        manifest['client_graph_indices'] = [list(idx) for idx in client_graph_indices]
    
    if metadata is not None:
        manifest['metadata'] = metadata
    
    # Create hash of the manifest for versioning
    manifest_str = json.dumps(manifest, sort_keys=True)
    manifest['hash'] = hashlib.sha256(manifest_str.encode()).hexdigest()
    
    return manifest

def save_manifest(manifest: Dict[str, Any], filepath: str):
    """Save manifest to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(manifest, f, indent=2)

def load_manifest(filepath: str) -> Dict[str, Any]:
    """Load manifest from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
