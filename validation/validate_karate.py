"""Self-contained validation on the Karate Club graph."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from fgl_heterogeneity.metrics import compute_full_heterogeneity_profile
from fgl_heterogeneity.partition import (
    BoundaryPolicy,
    apply_boundary_policy,
    community_split,
    dirichlet_label_split,
    ego_net_split,
    apply_feature_shift,
)
from fgl_heterogeneity.utils.datasets import load_karate_with_features
from fgl_heterogeneity.utils.manifest import generate_manifest, save_manifest


def _profile_from_node_sets(G, X, y, node_sets):
    graphs = [G.subgraph(nodes).copy() for nodes in node_sets]
    label_counts = [np.bincount(y[sorted(list(nodes))], minlength=len(np.unique(y))) for nodes in node_sets]
    embeddings = [X[sorted(list(nodes))] for nodes in node_sets]
    node_labels = [{node: int(y[node]) for node in nodes} for nodes in node_sets]
    global_adj = {node: set(G.neighbors(node)) for node in G.nodes()}
    sample_sizes = [len(nodes) for nodes in node_sets]
    return compute_full_heterogeneity_profile(
        label_counts=label_counts,
        embeddings=embeddings,
        graphs=graphs,
        node_labels=node_labels,
        node_sets=node_sets,
        global_adj=global_adj,
        sample_sizes=sample_sizes,
        subspace_rank=4,
    )


def main() -> None:
    out_dir = Path("outputs/karate_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    G, X, y = load_karate_with_features(seed=11)

    # P2 community split
    node_sets_comm, meta_comm = community_split(G, num_clients=4, resolution=1.0, seed=42)
    profile_comm = _profile_from_node_sets(G, X, y, node_sets_comm)
    save_manifest(generate_manifest(node_sets_comm, metadata={**meta_comm, "profile": profile_comm}), out_dir / "community_split.json")

    # P3 ego-net split
    node_sets_ego, meta_ego = ego_net_split(G, num_clients=4, anchors_per_client=2, hop_radius=1, seed=42, overlap_strategy="partial")
    profile_ego = _profile_from_node_sets(G, X, y, node_sets_ego)
    save_manifest(generate_manifest(node_sets_ego, metadata={**meta_ego, "profile": profile_ego}), out_dir / "egonet_split.json")

    # P1 applied to labels after a structural base partition
    dirichlet_indices, meta_dir = dirichlet_label_split(y, num_clients=4, alpha=0.2, seed=42)
    node_sets_dir = [set(idx.tolist()) for idx in dirichlet_indices]
    profile_dir = _profile_from_node_sets(G, X, y, node_sets_dir)
    save_manifest(generate_manifest(node_sets_dir, metadata={**meta_dir, "profile": profile_dir}), out_dir / "dirichlet_split.json")

    # P4 feature shift on top of community split
    base_embeddings = [X[sorted(list(nodes))] for nodes in node_sets_comm]
    shifted_embeddings, meta_shift = apply_feature_shift(base_embeddings, None, "affine", 0.35, seed=42)
    graphs = [G.subgraph(nodes).copy() for nodes in node_sets_comm]
    label_counts = [np.bincount(y[sorted(list(nodes))], minlength=len(np.unique(y))) for nodes in node_sets_comm]
    node_labels = [{node: int(y[node]) for node in nodes} for nodes in node_sets_comm]
    global_adj = {node: set(G.neighbors(node)) for node in G.nodes()}
    sample_sizes = [len(nodes) for nodes in node_sets_comm]
    profile_shift = compute_full_heterogeneity_profile(label_counts, shifted_embeddings, graphs, node_labels, node_sets_comm, global_adj, sample_sizes, subspace_rank=4)
    save_manifest(generate_manifest(node_sets_comm, metadata={**meta_shift, "base_protocol": "community_split", "profile": profile_shift}), out_dir / "community_feature_shift.json")

    # Boundary policy check
    _, boundary_meta = apply_boundary_policy(G, node_sets_comm, BoundaryPolicy.MASKED_BOUNDARY)
    with open(out_dir / "boundary_policy_summary.json", "w", encoding="utf-8") as f:
        json.dump(boundary_meta, f, indent=2)

    summary = {
        "community": profile_comm["domain_shift_profile"],
        "egonet": profile_ego["domain_shift_profile"],
        "dirichlet": profile_dir["domain_shift_profile"],
        "community_feature_shift": profile_shift["domain_shift_profile"],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Karate validation complete. Results saved to", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
