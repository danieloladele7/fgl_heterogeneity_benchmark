"""Cross-domain and quantity-skew validation on synthetic graph collections."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from fgl_heterogeneity.metrics import compute_full_heterogeneity_profile
from fgl_heterogeneity.partition import cross_domain_federation
from fgl_heterogeneity.utils.datasets import generate_synthetic_domain_collections
from fgl_heterogeneity.utils.manifest import _to_jsonable


def _graph_summary_embeddings(graphs):
    emb = []
    for G in graphs:
        emb.append(
            np.array(
                [
                    np.mean([d for _, d in G.degree()]),
                    nx.average_clustering(G),
                    G.number_of_nodes(),
                    G.number_of_edges(),
                ],
                dtype=float,
            )
        )
    return np.vstack(emb)


def main() -> None:
    out_dir = Path("outputs/synthetic_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = generate_synthetic_domain_collections(seed=19, graphs_per_domain=8)
    clients, meta = cross_domain_federation(domains, num_clients=2, assignment="one_per_client", seed=42)

    label_counts = []
    embeddings = []
    graphs = []
    node_labels = []
    node_sets = []
    sample_sizes = []

    for client in clients:
        payload = client["payloads"][0]
        client_graphs = payload["graphs"]
        client_graph_labels = np.array(payload["labels"], dtype=int)
        label_counts.append(np.bincount(client_graph_labels, minlength=2))
        embeddings.append(np.asarray(payload["features"], dtype=float))
        # aggregate a union graph only for structural descriptors in this simple PoC
        union_graph = nx.disjoint_union_all(client_graphs)
        graphs.append(union_graph)
        node_labels.append({n: int(n % 2) for n in union_graph.nodes()})
        node_sets.append(set(union_graph.nodes()))
        sample_sizes.append(len(client_graphs))

    global_adj = {idx: set(g.neighbors(idx)) for g in graphs for idx in g.nodes()}
    profile = compute_full_heterogeneity_profile(
        label_counts=label_counts,
        embeddings=embeddings,
        graphs=graphs,
        node_labels=node_labels,
        node_sets=node_sets,
        global_adj=global_adj,
        sample_sizes=sample_sizes,
        subspace_rank=2,
    )

    with open(out_dir / "cross_domain_profile.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable({"meta": meta, "profile": profile}), f, indent=2)

    print("Synthetic cross-domain validation complete. Results saved to", out_dir)
    print(json.dumps(profile["domain_shift_profile"], indent=2))


if __name__ == "__main__":
    main()
