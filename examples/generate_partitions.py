"""Generate manuscript-aligned example partitions on the Karate graph."""

from pathlib import Path

from fgl_heterogeneity.partition import community_split, dirichlet_label_split, ego_net_split
from fgl_heterogeneity.utils.datasets import load_karate_with_features
from fgl_heterogeneity.utils.manifest import generate_manifest, save_manifest


def main() -> None:
    out_dir = Path("outputs/example_splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    G, _, y = load_karate_with_features(seed=42)

    indices, meta_dir = dirichlet_label_split(y, num_clients=4, alpha=0.5, seed=42)
    save_manifest(generate_manifest(client_graph_indices=[idx.tolist() for idx in indices], metadata=meta_dir), out_dir / "dirichlet_alpha0.5.json")

    node_sets, meta_comm = community_split(G, num_clients=4, resolution=1.0, seed=42)
    save_manifest(generate_manifest(client_node_sets=node_sets, metadata=meta_comm), out_dir / "community_split.json")

    node_sets_ego, meta_ego = ego_net_split(G, num_clients=4, anchors_per_client=2, hop_radius=1, seed=42, overlap_strategy="partial")
    save_manifest(generate_manifest(client_node_sets=node_sets_ego, metadata=meta_ego), out_dir / "egonet_split.json")

    print("Saved example manifests to", out_dir)


if __name__ == "__main__":
    main()
