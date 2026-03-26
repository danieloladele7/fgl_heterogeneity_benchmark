"""Compute the recommended heterogeneity profile on a self-contained example."""

from fgl_heterogeneity.metrics import compute_full_heterogeneity_profile
from fgl_heterogeneity.partition import community_split
from fgl_heterogeneity.utils.datasets import load_karate_with_features
import numpy as np


def main() -> None:
    G, X, y = load_karate_with_features(seed=42)
    node_sets, _ = community_split(G, num_clients=4, seed=42)

    label_counts = [np.bincount(y[sorted(list(nodes))], minlength=len(np.unique(y))) for nodes in node_sets]
    embeddings = [X[sorted(list(nodes))] for nodes in node_sets]
    graphs = [G.subgraph(nodes).copy() for nodes in node_sets]
    node_labels = [{node: int(y[node]) for node in nodes} for nodes in node_sets]
    global_adj = {node: set(G.neighbors(node)) for node in G.nodes()}
    sample_sizes = [len(nodes) for nodes in node_sets]

    profile = compute_full_heterogeneity_profile(
        label_counts=label_counts,
        embeddings=embeddings,
        graphs=graphs,
        node_labels=node_labels,
        node_sets=node_sets,
        global_adj=global_adj,
        sample_sizes=sample_sizes,
        subspace_rank=4,
    )
    print(profile["domain_shift_profile"])


if __name__ == "__main__":
    main()
