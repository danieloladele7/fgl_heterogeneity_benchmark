"""
Example script to compute heterogeneity metrics.
"""
import networkx as nx
import numpy as np
from fgl_heterogeneity.metrics.aggregation import compute_full_heterogeneity_profile

# Load a pre-split federation (simulate client data)
def load_sample_federation():
    # Create synthetic clients
    num_clients = 3
    label_counts = []
    embeddings = []
    graphs = []
    node_labels = []
    node_sets = []
    sample_sizes = []
    
    for c in range(num_clients):
        # Simulate label distribution
        label_counts.append(np.random.dirichlet([1]*5) * 100)
        
        # Simulate embeddings
        emb = np.random.randn(50, 16)
        embeddings.append(emb)
        
        # Simulate graph (random)
        g = nx.erdos_renyi_graph(20, 0.1)
        graphs.append(g)
        node_labels.append({i: np.random.randint(0, 2) for i in range(20)})
        node_sets.append(set(range(20)))
        sample_sizes.append(50)
    
    global_adj = {}
    for g in graphs:
        for u, v in g.edges():
            global_adj.setdefault(u, set()).add(v)
            global_adj.setdefault(v, set()).add(u)
    
    return label_counts, embeddings, graphs, node_labels, node_sets, global_adj, sample_sizes

label_counts, embeddings, graphs, node_labels, node_sets, global_adj, sample_sizes = load_sample_federation()

profile = compute_full_heterogeneity_profile(
    label_counts=label_counts,
    embeddings=embeddings,
    graphs=graphs,
    node_labels=node_labels,
    node_sets=node_sets,
    global_adj=global_adj,
    sample_sizes=sample_sizes,
    subspace_rank=5
)

print("Heterogeneity Profile:")
print(f"Label JSD (mean): {profile['label_jsd']['mean_pairwise']:.4f}")
print(f"ECD (mean): {profile.get('ecd', {}).get('mean_pairwise', 0):.4f}")
print(f"Homophily variance: {profile.get('homophily_variance', 0):.4f}")
print(f"Topo Divergence (mean): {profile.get('topo_divergence', {}).get('mean_pairwise', 0):.4f}")
print(f"QII: {profile['qii']:.4f}")
print(f"Domain Shift Profile: {profile['domain_shift_profile']}")
