"""
Example script to generate partitions using different protocols.
"""
import networkx as nx
import numpy as np
from fgl_heterogeneity.partition import (
    dirichlet_label_split, community_split, egonet_split,
    feature_shift, boundary_policies
)
from fgl_heterogeneity.utils.manifest import generate_manifest, save_manifest

# Load a sample graph (e.g., Cora)
def load_cora():
    # Placeholder: load from torch_geometric or networkx
    G = nx.karate_club_graph()
    labels = np.random.randint(0, 2, size=G.number_of_nodes())
    return G, labels

G, labels = load_cora()

# 1. Dirichlet label split (for node classification)
indices, meta_dir = dirichlet_label_split(labels, num_clients=5, alpha=0.5, seed=42)
manifest_dir = generate_manifest(client_graph_indices=indices, metadata=meta_dir)
save_manifest(manifest_dir, 'splits/dirichlet_alpha0.5.json')

# 2. Community split
node_sets, meta_comm = community_split(G, num_clients=5, resolution=1.0, seed=42)
manifest_comm = generate_manifest(client_node_sets=node_sets, metadata=meta_comm)
save_manifest(manifest_comm, 'splits/community_louvain.json')

# 3. Ego-net split (with overlap)
node_sets_ego, meta_ego = egonet_split(G, num_clients=3, anchors_per_client=2, hop_radius=1, seed=42, overlap_strategy='partial')
manifest_ego = generate_manifest(client_node_sets=node_sets_ego, metadata=meta_ego)
save_manifest(manifest_ego, 'splits/egonet_partial.json')

print("All splits generated and saved.")
