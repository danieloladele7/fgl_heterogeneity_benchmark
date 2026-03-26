import numpy as np
import pytest
import networkx as nx

from fgl_heterogeneity.metrics import (
    aggregate_pairwise_metrics,
    compute_homophily,
    embedding_centroid_divergence,
    homophily_gap,
    jensen_shannon_divergence,
    label_distribution_divergence,
    missing_neighbor_ratio,
    neighbor_overlap_index,
    principal_subspace_divergence,
    quantity_imbalance_index,
    topological_divergence,
)


def test_jsd_identity():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert np.isclose(jensen_shannon_divergence(p, q), 0.0)


def test_jsd_maximum_binary():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert np.isclose(jensen_shannon_divergence(p, q), np.log(2))


def test_label_distribution_divergence():
    counts = [np.array([10, 0]), np.array([0, 10]), np.array([5, 5])]
    res = label_distribution_divergence(counts)
    assert res["pairwise_jsd"][0, 1] > res["pairwise_jsd"][0, 2]
    assert 0 <= res["lpv"] <= 0.25
    assert len(res["jsd_per_client"]) == 3


def test_ecd_and_psd_shapes():
    rng = np.random.default_rng(0)
    emb1 = rng.normal(size=(10, 5))
    emb2 = emb1 + 0.1 * rng.normal(size=(10, 5))
    ecd = embedding_centroid_divergence([emb1, emb2])
    psd = principal_subspace_divergence([emb1, emb2], rank=2)
    assert ecd["pairwise_ecd"].shape == (2, 2)
    assert psd["pairwise_psd"].shape == (2, 2)
    assert ecd["pairwise_ecd"][0, 1] >= 0
    assert psd["pairwise_psd"][0, 1] >= 0


def test_homophily_and_gap():
    G = nx.path_graph(3)
    labels_a = {0: 0, 1: 1, 2: 0}
    labels_b = {0: 0, 1: 0, 2: 0}
    h_a = compute_homophily(G, labels_a)
    h_b = compute_homophily(G, labels_b)
    assert 0 <= h_a <= 1
    res = homophily_gap([G, G], [labels_a, labels_b])
    assert np.isclose(res["pairwise_hg"][0, 1], abs(h_a - h_b))


def test_topological_divergence_zero_for_identical_graphs():
    G = nx.karate_club_graph()
    res = topological_divergence([G, G], metric="degree")
    assert np.isclose(res["pairwise_td"][0, 1], 0.0)


def test_overlap_and_missing_neighbor_metrics():
    node_sets = [set([1, 2]), set([2, 3])]
    global_adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    noi = neighbor_overlap_index(node_sets)
    assert np.isclose(noi["pairwise_noi"][0, 1], 1 / 3)
    mnr = missing_neighbor_ratio(node_sets, global_adj)
    assert np.isclose(mnr[0], 0.5)
    assert np.isclose(mnr[1], 0.5)


def test_quantity_imbalance():
    sizes = [10, 20, 30]
    res = quantity_imbalance_index(sizes)
    assert np.isclose(res["qii"], np.std(sizes) / np.mean(sizes))


def test_aggregate_pairwise_metrics():
    mat = np.array([[0, 0.5, 1.0], [0.5, 0, 0.8], [1.0, 0.8, 0]])
    res = aggregate_pairwise_metrics(mat)
    assert np.isclose(res["mean_pairwise"], np.mean([0.5, 1.0, 0.8]))
    assert np.isclose(res["max_pairwise"], 1.0)
