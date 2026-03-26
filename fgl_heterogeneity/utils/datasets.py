"""Self-contained dataset helpers for manuscript validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from networkx.generators.community import stochastic_block_model


@dataclass
class DomainGraphCollection:
    domain_id: int
    graphs: List[nx.Graph]
    labels: List[int]
    features: List[np.ndarray]


def load_karate_with_features(seed: int = 42) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Return the Karate Club graph with simple deterministic node features and labels."""
    rng = np.random.default_rng(seed)
    G = nx.karate_club_graph()
    labels = np.array([0 if G.nodes[n]["club"] == "Mr. Hi" else 1 for n in G.nodes()], dtype=int)
    degree = np.array([G.degree[n] for n in G.nodes()], dtype=float)
    clustering = np.array([nx.clustering(G, n) for n in G.nodes()], dtype=float)
    pagerank = np.array(list(nx.pagerank(G).values()), dtype=float)
    noise = rng.normal(scale=0.05, size=(G.number_of_nodes(), 3))
    X = np.column_stack([degree, clustering, pagerank, noise])
    return G, X, labels


def generate_synthetic_domain_collections(
    seed: int = 42,
    graphs_per_domain: int = 8,
) -> List[Dict]:
    """Generate two graph collections with controlled cross-domain shift.

    Domain 0: homophilic SBM graphs.
    Domain 1: lower-homophily / more irregular BA-style graphs.
    """
    rng = np.random.default_rng(seed)
    collections = []

    # Domain 0: SBM, balanced labels, strong communities
    domain0_graphs = []
    domain0_labels = []
    domain0_features = []
    for _ in range(graphs_per_domain):
        sizes = [15, 15]
        probs = [[0.28, 0.03], [0.03, 0.26]]
        G = stochastic_block_model(sizes, probs, seed=int(rng.integers(1e9)))
        y = int(rng.integers(0, 2))
        feat = np.array([
            np.mean([d for _, d in G.degree()]),
            nx.average_clustering(G),
            G.number_of_nodes(),
            G.number_of_edges(),
        ], dtype=float)
        domain0_graphs.append(nx.convert_node_labels_to_integers(G))
        domain0_labels.append(y)
        domain0_features.append(feat)
    collections.append({
        "domain_id": 0,
        "payload": {
            "graphs": domain0_graphs,
            "labels": domain0_labels,
            "features": domain0_features,
        },
    })

    # Domain 1: BA graphs with noisier feature profile
    domain1_graphs = []
    domain1_labels = []
    domain1_features = []
    for _ in range(graphs_per_domain):
        G = nx.barabasi_albert_graph(30, 2, seed=int(rng.integers(1e9)))
        y = int(rng.integers(0, 2))
        feat = np.array([
            np.mean([d for _, d in G.degree()]),
            nx.average_clustering(G),
            G.number_of_nodes(),
            G.number_of_edges(),
        ], dtype=float) + np.array([0.4, -0.08, 0.0, 5.0])
        domain1_graphs.append(nx.convert_node_labels_to_integers(G))
        domain1_labels.append(y)
        domain1_features.append(feat)
    collections.append({
        "domain_id": 1,
        "payload": {
            "graphs": domain1_graphs,
            "labels": domain1_labels,
            "features": domain1_features,
        },
    })

    return collections
