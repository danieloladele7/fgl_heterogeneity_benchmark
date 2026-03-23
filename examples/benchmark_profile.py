"""
Full example: load a dataset, generate partitions using different protocols,
compute heterogeneity profile, and save results.
"""

import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

from fgl_heterogeneity.partition.dirichlet_split import dirichlet_label_split
from fgl_heterogeneity.partition.community_split import community_split
from fgl_heterogeneity.partition.egonet_split import ego_net_split
from fgl_heterogeneity.partition.crossdomain_split import cross_domain_federation
from fgl_heterogeneity.partition.feature_shift import apply_feature_shift
from fgl_heterogeneity.partition.boundary_policies import BoundaryPolicy, apply_boundary_policy
from fgl_heterogeneity.metrics.aggregation import compute_full_heterogeneity_profile
from fgl_heterogeneity.utils.io_utils import save_split_manifest, ensure_directory


def load_cora():
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    # node features
    x = data.x.numpy()
    labels = data.y.numpy()
    return G, x, labels


def main():
    # Create output directory
    out_dir = './benchmark_results'
    ensure_directory(out_dir)

    # Load graph
    G, x, labels = load_cora()
    n_nodes = G.number_of_nodes()
    n_classes = len(np.unique(labels))

    # --- 1. Dirichlet label split (only label skew, no structural partition)
    # For node classification, Dirichlet split is applied to labeled nodes.
    # Here we use all nodes as labeled, but in practice you might have a train/test split.
    # We'll just get node indices for each class.
    node_indices = np.arange(n_nodes)
    class_indices = [node_indices[labels == c] for c in range(n_classes)]
    # Convert to per-node label array
    label_array = labels  # shape (n_nodes,)
    indices, meta_dir = dirichlet_label_split(label_array, num_clients=5, alpha=0.5, seed=42)
    # Convert indices to node sets (since each client gets a set of nodes)
    dirichlet_node_sets = [set(idx) for idx in indices]

    # Create induced subgraphs for each client
    dirichlet_subgraphs = [G.subgraph(s) for s in dirichlet_node_sets]

    # Compute metrics for Dirichlet split
    # For label counts, we need per-client class counts. We can compute from the node sets.
    label_counts = [np.bincount(labels[list(s)], minlength=n_classes) for s in dirichlet_node_sets]
    # For embeddings, we need per-node embeddings. Use node features as embeddings.
    embeddings = [x[list(s)] for s in dirichlet_node_sets]
    # For homophily, we need node labels per client
    client_labels = [{node: int(labels[node]) for node in s} for s in dirichlet_node_sets]
    sample_sizes = [len(s) for s in dirichlet_node_sets]

    profile_dir = compute_full_heterogeneity_profile(
        label_counts=label_counts,
        embeddings=embeddings,
        graphs=dirichlet_subgraphs,
        node_labels=client_labels,
        node_sets=dirichlet_node_sets,
        global_adj={node: set(G.neighbors(node)) for node in G.nodes()},
        sample_sizes=sample_sizes,
        subspace_rank=10
    )
    print("Dirichlet split profile:")
    print(profile_dir['domain_shift_profile'])
    # Save manifest
    manifest_dir = {
        'protocol': 'dirichlet_label_split',
        'num_clients': 5,
        'alpha': 0.5,
        'seed': 42,
        'client_node_sets': [list(s) for s in dirichlet_node_sets],
        'profile': profile_dir
    }
    save_split_manifest(manifest_dir, os.path.join(out_dir, 'dirichlet_alpha0.5.json'))

    # --- 2. Community split
    client_node_sets, meta_comm = community_split(G, num_clients=5, resolution=1.0, seed=42)
    comm_subgraphs = [G.subgraph(s) for s in client_node_sets]
    label_counts_comm = [np.bincount(labels[list(s)], minlength=n_classes) for s in client_node_sets]
    embeddings_comm = [x[list(s)] for s in client_node_sets]
    client_labels_comm = [{node: int(labels[node]) for node in s} for s in client_node_sets]
    sample_sizes_comm = [len(s) for s in client_node_sets]

    profile_comm = compute_full_heterogeneity_profile(
        label_counts=label_counts_comm,
        embeddings=embeddings_comm,
        graphs=comm_subgraphs,
        node_labels=client_labels_comm,
        node_sets=client_node_sets,
        global_adj={node: set(G.neighbors(node)) for node in G.nodes()},
        sample_sizes=sample_sizes_comm,
        subspace_rank=10
    )
    print("Community split profile:")
    print(profile_comm['domain_shift_profile'])
    manifest_comm = {
        'protocol': 'community_split',
        'num_clients': 5,
        'resolution': 1.0,
        'seed': 42,
        'client_node_sets': [list(s) for s in client_node_sets],
        'profile': profile_comm
    }
    save_split_manifest(manifest_comm, os.path.join(out_dir, 'community_louvain.json'))

    # --- 3. Ego-net split
    client_node_sets_ego, meta_ego = ego_net_split(G, num_clients=5, anchors_per_client=5,
                                                   hop_radius=1, seed=42, overlap_strategy='partial')
    ego_subgraphs = [G.subgraph(s) for s in client_node_sets_ego]
    label_counts_ego = [np.bincount(labels[list(s)], minlength=n_classes) for s in client_node_sets_ego]
    embeddings_ego = [x[list(s)] for s in client_node_sets_ego]
    client_labels_ego = [{node: int(labels[node]) for node in s} for s in client_node_sets_ego]
    sample_sizes_ego = [len(s) for s in client_node_sets_ego]

    profile_ego = compute_full_heterogeneity_profile(
        label_counts=label_counts_ego,
        embeddings=embeddings_ego,
        graphs=ego_subgraphs,
        node_labels=client_labels_ego,
        node_sets=client_node_sets_ego,
        global_adj={node: set(G.neighbors(node)) for node in G.nodes()},
        sample_sizes=sample_sizes_ego,
        subspace_rank=10
    )
    print("Ego-net split profile:")
    print(profile_ego['domain_shift_profile'])
    manifest_ego = {
        'protocol': 'egonet_split',
        'num_clients': 5,
        'anchors_per_client': 5,
        'hop_radius': 1,
        'seed': 42,
        'overlap_strategy': 'partial',
        'client_node_sets': [list(s) for s in client_node_sets_ego],
        'profile': profile_ego
    }
    save_split_manifest(manifest_ego, os.path.join(out_dir, 'egonet_partial.json'))

    # --- 4. Cross-domain (simulate using random splits of features?)
    # For demonstration, we create two "domains" by perturbing features of some nodes.
    # In practice, you'd load different datasets.
    # We'll split the graph into two domains based on a random partition.
    domain_mask = np.random.rand(n_nodes) < 0.5
    domain1_nodes = set(np.where(domain_mask)[0])
    domain2_nodes = set(np.where(~domain_mask)[0])
    # Create two domain datasets: each is a subgraph of the original.
    domain1_data = {'nodes': domain1_nodes, 'graph': G.subgraph(domain1_nodes), 'x': x[list(domain1_nodes)], 'labels': labels[list(domain1_nodes)]}
    domain2_data = {'nodes': domain2_nodes, 'graph': G.subgraph(domain2_nodes), 'x': x[list(domain2_nodes)], 'labels': labels[list(domain2_nodes)]}
    domains = [domain1_data, domain2_data]
    # Cross-domain federation: assign each domain to a client (one_per_client)
    client_data, meta_cd = cross_domain_federation(domains, num_clients=2, assignment='one_per_client', seed=42)
    # For metrics, we need node sets, graphs, etc. per client
    cd_node_sets = [data['nodes'] for data in client_data]
    cd_subgraphs = [data['graph'] for data in client_data]
    cd_label_counts = [np.bincount(data['labels'], minlength=n_classes) for data in client_data]
    cd_embeddings = [data['x'] for data in client_data]
    cd_client_labels = [{node: int(labels[node]) for node in data['nodes']} for data in client_data]
    cd_sample_sizes = [len(data['nodes']) for data in client_data]

    profile_cd = compute_full_heterogeneity_profile(
        label_counts=cd_label_counts,
        embeddings=cd_embeddings,
        graphs=cd_subgraphs,
        node_labels=cd_client_labels,
        node_sets=cd_node_sets,
        global_adj={node: set(G.neighbors(node)) for node in G.nodes()},
        sample_sizes=cd_sample_sizes,
        subspace_rank=10
    )
    print("Cross-domain profile:")
    print(profile_cd['domain_shift_profile'])
    manifest_cd = {
        'protocol': 'crossdomain_split',
        'assignment': 'one_per_client',
        'seed': 42,
        'client_node_sets': [list(s) for s in cd_node_sets],
        'profile': profile_cd
    }
    save_split_manifest(manifest_cd, os.path.join(out_dir, 'crossdomain.json'))

    # --- 5. Feature shift on top of community split
    # Apply feature shift to the community split clients
    transformed_features, meta_fs = apply_feature_shift(
        feature_matrices=embeddings_comm,
        client_assignments=None,
        shift_type='affine',
        shift_intensity=0.5,
        seed=42
    )
    # Compute profile with shifted features
    profile_fs = compute_full_heterogeneity_profile(
        label_counts=label_counts_comm,
        embeddings=transformed_features,
        graphs=comm_subgraphs,
        node_labels=client_labels_comm,
        node_sets=client_node_sets,
        global_adj={node: set(G.neighbors(node)) for node in G.nodes()},
        sample_sizes=sample_sizes_comm,
        subspace_rank=10
    )
    print("Community + feature shift profile:")
    print(profile_fs['domain_shift_profile'])
    manifest_fs = {
        'protocol': 'community_split + feature_shift',
        'shift_type': 'affine',
        'shift_intensity': 0.5,
        'client_node_sets': [list(s) for s in client_node_sets],
        'profile': profile_fs
    }
    save_split_manifest(manifest_fs, os.path.join(out_dir, 'community_feature_shift.json'))

    print("\nAll benchmarks saved in 'benchmark_results' directory.")


if __name__ == '__main__':
    main()
