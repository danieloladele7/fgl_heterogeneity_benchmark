import numpy as np
import networkx as nx

from fgl_heterogeneity.partition import (
    BoundaryPolicy,
    apply_boundary_policy,
    apply_feature_shift,
    community_split,
    cross_domain_federation,
    dirichlet_label_split,
    ego_net_split,
)


def test_dirichlet_split_assigns_every_item_once():
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 1, 0])
    indices, meta = dirichlet_label_split(labels, num_clients=3, alpha=0.5, seed=42)
    concatenated = np.concatenate(indices)
    assert len(concatenated) == len(labels)
    assert len(set(concatenated.tolist())) == len(labels)
    assert meta["per_client_counts"] == [len(idx) for idx in indices]


def test_community_split_covers_all_nodes_once():
    G = nx.karate_club_graph()
    node_sets, meta = community_split(G, num_clients=3, seed=42)
    union = set().union(*node_sets)
    assert union == set(G.nodes())
    assert sum(len(s) for s in node_sets) == G.number_of_nodes()
    assert meta["num_clients"] == 3


def test_egonet_monotonicity_with_hop_radius():
    G = nx.erdos_renyi_graph(50, 0.05, seed=42)
    node_sets_r1, _ = ego_net_split(G, num_clients=5, anchors_per_client=2, hop_radius=1, seed=42, overlap_strategy="partial")
    node_sets_r2, _ = ego_net_split(G, num_clients=5, anchors_per_client=2, hop_radius=2, seed=42, overlap_strategy="partial")
    for s1, s2 in zip(node_sets_r1, node_sets_r2):
        assert s1.issubset(s2)


def test_cross_domain_one_to_many_covers_all_domains():
    datasets = [{"domain_id": i, "payload": f"domain_{i}"} for i in range(3)]
    clients, meta = cross_domain_federation(datasets, num_clients=2, assignment="one_to_many", seed=42)
    covered = sorted(set(d for c in clients for d in c["domain_ids"]))
    assert covered == [0, 1, 2]
    assert meta["assignment"] == "one_to_many"


def test_feature_shift_changes_features_only():
    feats = [np.random.randn(10, 5), np.random.randn(10, 5)]
    transformed, meta = apply_feature_shift(feats, None, "gaussian_noise", 0.1, seed=42)
    assert transformed[0].shape == feats[0].shape
    assert not np.allclose(transformed[0], feats[0])
    assert meta["shift_type"] == "gaussian_noise"


def test_boundary_policy_masked_returns_counts():
    G = nx.cycle_graph(5)
    node_sets = [set([0, 1, 2]), set([2, 3, 4])]
    subgraphs, meta = apply_boundary_policy(G, node_sets, BoundaryPolicy.MASKED_BOUNDARY)
    assert len(subgraphs) == 2
    assert "boundary_counts" in meta
    assert len(meta["boundary_counts"]) == 2


def test_boundary_policy_induced_drops_cross_edges():
    G = nx.cycle_graph(5)
    node_sets = [set([0, 1, 2])]
    subgraphs, _ = apply_boundary_policy(G, node_sets, BoundaryPolicy.INDUCED_SUBGRAPH)
    assert subgraphs[0].number_of_edges() == 2
