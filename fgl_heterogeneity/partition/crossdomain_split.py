"""
Cross-domain federation protocol as described in Section V-E of the manuscript.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Union


def cross_domain_federation(
    datasets: List[Any],
    num_clients: int,
    assignment: str = 'one_per_client',
    seed: int = 42,
    domain_ids: List[int] = None,
    mixing_ratios: List[float] = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Create a cross-domain federation by assigning domain-specific data to clients.

    Args:
        datasets: List of domain data objects. Each element can be a graph collection,
                  a list of graphs, or any object representing that domain.
        num_clients: Number of clients.
        assignment: Type of assignment:
            - 'one_per_client': each domain assigned to a single client (requires num_clients >= len(datasets))
            - 'one_to_many': each domain assigned to multiple clients (shards)
            - 'mixed': controlled mixing using mixing_ratios
        seed: Random seed.
        domain_ids: If assignment is 'mixed', list of domain IDs for each client.
        mixing_ratios: If assignment is 'mixed', list of mixing ratios (probabilities) per client.

    Returns:
        client_datasets: List of client data objects. Each is a combination of domain data.
        metadata: Dict with assignment info.
    """
    np.random.seed(seed)
    num_domains = len(datasets)

    if assignment == 'one_per_client':
        if num_clients < num_domains:
            raise ValueError(f"num_clients ({num_clients}) must be >= number of domains ({num_domains}) for one_per_client assignment.")
        client_datasets = [None] * num_clients
        # Assign each domain to a distinct client
        for i, ds in enumerate(datasets):
            client_datasets[i] = ds
        # For remaining clients, assign a copy of some domain (or empty)
        for j in range(num_domains, num_clients):
            # assign to a random existing domain (or leave None)
            client_datasets[j] = datasets[np.random.randint(0, num_domains)]

    elif assignment == 'one_to_many':
        # Each domain may be split among several clients
        client_datasets = [None] * num_clients
        for i in range(num_clients):
            # Randomly select a domain for this client
            domain_idx = np.random.randint(0, num_domains)
            client_datasets[i] = datasets[domain_idx]
        # Alternatively, could split each domain into shards; here we just assign full domains.
        # A more advanced split would partition domain data among clients.

    elif assignment == 'mixed':
        if domain_ids is None or mixing_ratios is None:
            raise ValueError("For 'mixed' assignment, domain_ids and mixing_ratios must be provided.")
        client_datasets = []
        for k in range(num_clients):
            # Build a mixture of domains according to mixing_ratios[k]
            mixture = {}
            for d, ratio in zip(domain_ids[k], mixing_ratios[k]):
                # Could sample ratio * n_samples from domain d
                # For simplicity, we return a reference and metadata
                mixture[datasets[d]] = ratio
            client_datasets.append(mixture)
    else:
        raise ValueError(f"Unknown assignment: {assignment}")

    metadata = {
        'protocol': 'crossdomain_split',
        'assignment': assignment,
        'num_clients': num_clients,
        'num_domains': num_domains,
        'seed': seed,
    }
    if mixing_ratios is not None:
        metadata['mixing_ratios'] = mixing_ratios
    if domain_ids is not None:
        metadata['domain_ids'] = domain_ids

    return client_datasets, metadata


def cross_domain_federation_with_sharding(
    datasets: List[Any],
    num_clients: int,
    samples_per_domain: List[int],
    seed: int = 42
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Alternative: shard each domain across clients based on sample counts.

    Args:
        datasets: List of domain data objects, each expected to be indexable.
        num_clients: Number of clients.
        samples_per_domain: Number of samples to assign from each domain to each client? 
                            This is a simplified version; actual sharding may be more complex.
        seed: Random seed.
    """
    np.random.seed(seed)
    num_domains = len(datasets)
    client_data = [[] for _ in range(num_clients)]

    for d_idx, domain_data in enumerate(datasets):
        # assume domain_data is a list of items (graphs, etc.)
        n_domain = len(domain_data)
        # Shuffle domain data
        shuffled_indices = np.random.permutation(n_domain)
        # Split into num_clients shards
        shard_size = n_domain // num_clients
        for k in range(num_clients):
            start = k * shard_size
            end = (k+1) * shard_size if k < num_clients-1 else n_domain
            shard = [domain_data[i] for i in shuffled_indices[start:end]]
            client_data[k].extend(shard)

    metadata = {
        'protocol': 'crossdomain_split_with_sharding',
        'num_clients': num_clients,
        'seed': seed,
    }
    return client_data, metadata
