# FGL Heterogeneity Benchmark

A reproducible benchmarking suite for non-IID data in Federated Graph Learning, implementing the taxonomy, metrics, and partition protocols from "Benchmarking Non-IID Data in Federated Graph Learning".

## Features

- **8 metric types** covering label skew, feature skew, topology skew, overlap, quantity skew, and domain shift.
- **5 partition protocols** with deterministic reproducibility.
- **Boundary-edge policies** for subgraph federations.
- **Manifest generation** for auditability and version control.

## Installation

```bash
git clone https://github.com/your-org/fgl-heterogeneity-benchmark.git
cd fgl-heterogeneity-benchmark
pip install -e .
```

## Quick Start

See `examples/` for usage.

## Citation

If you use this code, please cite the accompanying manuscript.

```text


```

## License

MIT

### Testing (Optional)

```python
# tests/test_metrics.py
import numpy as np
from fgl_heterogeneity.metrics.label_metrics import jensen_shannon_divergence, label_distribution_divergence

def test_jsd():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert jensen_shannon_divergence(p, q) == 0.0

    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    jsd = jensen_shannon_divergence(p, q)
    assert np.isclose(jsd, np.log(2))  # maximum JSD = log(2)

def test_label_divergence():
    counts = [np.array([10, 0]), np.array([0, 10]), np.array([5, 5])]
    res = label_distribution_divergence(counts)
    assert res['pairwise_jsd'][0, 1] > res['pairwise_jsd'][0, 2]
```
