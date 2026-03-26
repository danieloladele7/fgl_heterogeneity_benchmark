"""Protocol invariant checks for manuscript validation."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from fgl_heterogeneity.partition import (
    BoundaryPolicy,
    apply_boundary_policy,
    community_split,
    cross_domain_federation,
    dirichlet_label_split,
    ego_net_split,
    apply_feature_shift,
)
from fgl_heterogeneity.utils.datasets import (
    generate_synthetic_domain_collections,
    load_karate_with_features,
)
from fgl_heterogeneity.utils.manifest import generate_manifest


def manifest_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> None:
    G, X, y = load_karate_with_features(seed=7)

    # P1 invariants
    split1, meta1 = dirichlet_label_split(y, num_clients=4, alpha=0.5, seed=123)
    split1b, meta1b = dirichlet_label_split(y, num_clients=4, alpha=0.5, seed=123)
    assert sum(len(s) for s in split1) == len(y)
    assert len(set(np.concatenate(split1).tolist())) == len(y)
    assert [s.tolist() for s in split1] == [s.tolist() for s in split1b]
    assert meta1 == meta1b

    # P2 invariants
    comm, meta_comm = community_split(G, num_clients=4, seed=123)
    all_nodes = set().union(*comm)
    assert all_nodes == set(G.nodes())
    assert sum(len(s) for s in comm) == G.number_of_nodes()
    comm2, _ = community_split(G, num_clients=4, seed=123)
    assert comm == comm2

    # P3 invariants
    ego1, meta_ego1 = ego_net_split(G, num_clients=4, anchors_per_client=2, hop_radius=1, seed=123, overlap_strategy="partial")
    ego2, _ = ego_net_split(G, num_clients=4, anchors_per_client=2, hop_radius=2, seed=123, overlap_strategy="partial")
    for s1, s2 in zip(ego1, ego2):
        assert s1.issubset(s2)
    assert all(set(meta_ego1["anchors"][i]).issubset(set(G.nodes())) for i in range(4))

    # P4 invariants
    node_sets, _ = community_split(G, num_clients=4, seed=123)
    base_features = [X[sorted(list(s))] for s in node_sets]
    shifted, _ = apply_feature_shift(base_features, None, "gaussian_noise", 0.1, seed=123)
    for base, out in zip(base_features, shifted):
        assert base.shape == out.shape
        assert not np.allclose(base, out)

    # P5 invariants
    domains = generate_synthetic_domain_collections(seed=7, graphs_per_domain=4)
    clients, meta_cd = cross_domain_federation(domains, num_clients=2, assignment="one_to_many", seed=123)
    covered = sorted(set(d for client in clients for d in client["domain_ids"]))
    assert covered == [0, 1]
    clients_b, meta_cd_b = cross_domain_federation(domains, num_clients=2, assignment="one_to_many", seed=123)
    assert clients == clients_b
    assert meta_cd == meta_cd_b

    # Boundary policies
    induced, meta_induced = apply_boundary_policy(G, comm, BoundaryPolicy.INDUCED_SUBGRAPH)
    masked, meta_masked = apply_boundary_policy(G, comm, BoundaryPolicy.MASKED_BOUNDARY)
    assert len(induced) == len(masked) == 4
    assert len(meta_masked["boundary_counts"]) == 4
    assert len(meta_induced["boundary_counts"]) == 4

    # Manifest determinism
    man = generate_manifest(client_node_sets=comm, metadata=meta_comm)
    man2 = generate_manifest(client_node_sets=comm2, metadata=meta_comm)
    assert man["hash"] == man2["hash"]

    print("All protocol invariant checks passed.")


if __name__ == "__main__":
    main()
