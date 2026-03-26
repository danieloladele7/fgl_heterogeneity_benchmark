# Validation summary

This repository was patched into a manuscript-ready proof-of-concept benchmark for validating the taxonomy, metrics, and partition protocols in the accompanying federated graph learning review.

## What was fixed

- repaired package-level imports and exports
- removed the hard dependency on `python-louvain` by using `networkx.community.louvain_communities`
- made topology divergence use shared histogram bins across clients
- clarified ego-net overlap semantics so that `disjoint` means disjoint anchors rather than guaranteed disjoint node sets
- made cross-domain federation deterministic and coverage-complete
- exposed masked-boundary and induced-subgraph metadata through boundary-policy outputs
- hardened Dirichlet splitting so empty clients are avoided in small-data settings
- added self-contained validation scripts and notebook
- rewrote tests to check protocol invariants rather than incidental behavior

## Datasets used in the included proof of concept

To keep the benchmark runnable without external downloads, the repository validates on:

1. **Karate Club graph** for node/subgraph protocol checks
2. **Synthetic two-domain graph collections** for cross-domain and many-graph checks

A `torch_geometric` loader remains available for later expansion to datasets such as Cora.

## Validation workflow

Run:

```bash
python -m validation.run_all_validation
```

This executes:

1. protocol invariant checks
2. Karate-based validation
3. synthetic cross-domain validation

## Current status

- `pytest -q tests validation/test_protocol_invariants.py` passes
- `python -m validation.run_all_validation` completes successfully

## Output locations

- `outputs/karate_validation/`
- `outputs/synthetic_validation/`

These contain manifests, metric summaries, and protocol metadata suitable for appendix-level auditability.
