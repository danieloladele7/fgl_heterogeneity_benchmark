import numpy as np
import pytest
from fgl_heterogeneity.metrics.label_metrics import jensen_shannon_divergence, label_distribution_divergence
from fgl_heterogeneity.metrics.representation_metrics import embedding_centroid_divergence, principal_subspace_divergence
from fgl_heterogeneity.metrics.topology_metrics import compute_homophily, homophily_gap, topological_divergence
from fgl_heterogeneity.metrics.overlap_metrics import neighbor_overlap_index, missing_neighbor_ratio
from fgl_heterogeneity.metrics.quantity_metrics import quantity_imbalance_index
from fgl_heterogeneity.metrics.aggregation import aggregate_pairwise_metrics


class TestLabelMetrics:
    def test_jsd_identical(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert jensen_shannon_divergence(p, q) == 0.0

    def test_jsd_maximum(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert np.isclose(jsd, np.log(2))

    def test_label_distribution_divergence(self):
        counts = [np.array([10, 0]), np.array([0, 10]), np.array([5, 5])]
        res = label_distribution_divergence(counts)
        assert res['pairwise_jsd'][0, 1] > res['pairwise_jsd'][0, 2]
        assert 0 <= res['lpv'] <= 0.25  # max variance for binary
        assert len(res['jsd_per_client']) == 3


class TestRepresentationMetrics:
    def test_ecd(self):
        emb1 = np.random.randn(10, 5)
        emb2 = np.random.randn(10, 5)
        emb_list = [emb1, emb2]
        res = embedding_centroid_divergence(emb_list)
        assert res['pairwise_ecd'].shape == (2, 2)
        assert res['pairwise_ecd'][0, 1] >= 0

    def test_psd(self):
        emb1 = np.random.randn(10, 5)
        emb2 = emb1 + 0.1 * np.random.randn(10, 5)  # similar
        res = principal_subspace_divergence([emb1, emb2], rank=2)
        assert res['pairwise_psd'].shape == (2, 2)
        assert res['pairwise_psd'][0, 1] < 1.0


class TestTopologyMetrics:
    def test_homophily(self, simple_graph):
        g, labels = simple_graph
        h = compute_homophily(g, labels)
        assert 0 <= h <= 1

    def test_homophily_gap(self, simple_graph):
        g, labels = simple_graph
        graphs = [g, g]
        labels_list = [labels, labels]
        res = homophily_gap(graphs, labels_list)
        assert res['pairwise_hg'][0, 1] == 0.0
        assert len(res['homophily_per_client']) == 2

    def test_topological_divergence(self, simple_graph):
        g, _ = simple_graph
        graphs = [g, g]
        res = topological_divergence(graphs, metric='degree')
        assert res['pairwise_td'][0, 1] == 0.0


class TestOverlapMetrics:
    def test_neighbor_overlap_index(self):
        sets = [set([1, 2, 3]), set([2, 3, 4]), set([5, 6])]
        res = neighbor_overlap_index(sets)
        assert res['pairwise_noi'][0, 1] > 0
        assert res['pairwise_noi'][0, 2] == 0

    def test_missing_neighbor_ratio(self):
        node_sets = [set([1, 2]), set([2, 3])]
        global_adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
        mnr = missing_neighbor_ratio(node_sets, global_adj)
        # For set {1,2}, missing neighbors: 1->3, 2->3 -> 2 missing out of total deg 2+2=4 -> 0.5
        assert mnr[0] == 0.5
        # For {2,3}, missing neighbors: 2->1,3->1 -> 2 missing out of total deg 2+2=4 -> 0.5
        assert mnr[1] == 0.5


class TestQuantityMetrics:
    def test_qii(self):
        sizes = [10, 20, 30]
        res = quantity_imbalance_index(sizes)
        assert res['qii'] == np.std(sizes) / np.mean(sizes)
        assert res['per_client'] == sizes


class TestAggregation:
    def test_aggregate_pairwise(self):
        mat = np.array([[0, 0.5, 1.0], [0.5, 0, 0.8], [1.0, 0.8, 0]])
        res = aggregate_pairwise_metrics(mat)
        assert res['mean_pairwise'] == np.mean([0.5, 1.0, 0.8])
        assert res['max_pairwise'] == 1.0
        assert res['per_client_mean'][0] == (0.5 + 1.0) / 2
        assert res['per_client_mean'][1] == (0.5 + 0.8) / 2
        assert res['per_client_mean'][2] == (1.0 + 0.8) / 2


# Fixture
@pytest.fixture
def simple_graph():
    import networkx as nx
    G = nx.path_graph(3)
    labels = {0: 0, 1: 1, 2: 0}
    return G, labels
