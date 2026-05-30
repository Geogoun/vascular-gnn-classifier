# Vascular Disease Classification from Sparse Flow Measurements

Physics-informed graph neural networks for classifying vascular pathologies using partial hemodynamic observations. The model generalizes across vascular anatomies — trained on one network, it transfers to a structurally different network.

## Problem

In real microvascular imaging, you observe blood flow in a small fraction of vessels. Can you infer the disease state of the whole network from these sparse, noisy measurements?

We generate synthetic pathology data using a first-principles Hagen–Poiseuille flow solver on real vascular graph topologies, then train classifiers to distinguish three conditions:

| Class | Mechanism | Flow signature |
|---|---|---|
| **Healthy** | Small random diameter jitter | Within measurement noise |
| **Boundary blockage** | Narrowing at a boundary node | Distributed shifts across downstream edges |
| **Regional damage** | Narrowing within a local neighborhood | Localized large changes in one region |

## Approach

### Physics solver

A graph Laplacian flow solver derived from Hagen–Poiseuille law and mass conservation computes steady-state flows given vessel diameters, lengths, and boundary conditions. This solver generates all training data — it is not called at inference.

### Three classifier architectures

All three produce a graph-level class prediction from 6-dimensional edge features (observed flow, observation mask, healthy baseline flow, log-diameter, log-length, log-conductance):

- **Per-edge MLP** — shared MLP per edge → mean+max pool → classifier head. No graph structure.
- **Flat MLP** — concatenate all edge features → MLP → classifier head. No graph structure, no weight sharing. Cannot transfer across networks with different edge counts.
- **Edge GNN** — message passing on the vascular graph → mean+max pool → classifier head. Exploits both graph topology and shared weights.

### Cross-anatomy transfer

The key experiment: train on **Network 2** (284 nodes, 399 edges) and test on **Network 1** (395 nodes, 553 edges). The GNN and per-edge MLP architectures transfer because they use shared weights; the flat MLP cannot (fixed input dimension).

### Physics-informed feature ablation

Configurable weights control how strongly physics-derived features (healthy baseline flow, vessel geometry) enter the classifier. Sweeping these weights reveals how much the physics prior helps or hurts cross-anatomy generalization.

## Implementation

Built entirely in **JAX/Flax** with:
- Differentiable Hagen–Poiseuille solver (graph Laplacian + `jnp.linalg.solve`)
- `vmap`-ed training over batches of sparse observations
- `optax` (Adam) with early stopping
- Flax `linen` modules for all three architectures

## Data

Two real microvascular network topologies extracted from experimental imaging data courtesy of T.W. Secomb, Department of Physiology,
University of Arizona. Original data and additional networks available at:
https://sites.arizona.edu/secomb/microvascular-networks-3d-structural-information/

```
data/
├── network1/          # 395 nodes, 553 edges
│   ├── 28_10_90_node_data_thresh0.csv
│   ├── 28_10_90_segment_data_thresh0.csv
│   └── 28_10_90_boundary_data_thresh0.csv
└── network2/          # 284 nodes, 399 edges
    ├── 15_02_84_node_data_thresh0.csv
    ├── 15_02_84_segment_data_thresh0.csv
    └── 15_02_84_boundary_data_thresh0.csv
```

Each network is defined by node coordinates, segment connectivity (with diameters and lengths), and boundary flow conditions.

## Quick start

```bash
git clone https://github.com/Geogoun/vascular-gnn-classifier.git
cd vascular-gnn-classifier
pip install -r requirements.txt
jupyter notebook disease_classification.ipynb
```

The notebook runs end-to-end: data generation → training → evaluation → visualization. On a CPU, expect ~5–10 minutes for the full pipeline. GPU (via JAX) speeds up training significantly.

## Configuration

All experiment parameters are set in the `CONFIG` dictionary at the top of the notebook:

| Parameter | Default | Description |
|---|---|---|
| `train_network` | 2 | Which network to train on |
| `test_network` | 1 | Which network to evaluate on (cross-anatomy) |
| `observation_fraction` | 0.2 | Fraction of edges observed per sample |
| `K_dataset_net*` | 6000 | Samples per network |
| `physics_flow_feature_weight` | 1.0 | Weight of physics-flow prior (0 = ablated) |
| `n_rounds` | 3 | GNN message-passing rounds |

## What to look for in the results

1. **Architecture ranking**: if `edge_gnn` > `per_edge_mlp` > `mlp`, message passing helps detect spatial damage patterns.
2. **Regional damage class**: this is where the GNN should win — the localized pattern requires multi-hop information aggregation.
3. **Cross-anatomy transfer**: the flat MLP cannot transfer at all (fixed input size), providing a strong baseline argument for graph-based approaches.

## Requirements

- Python ≥ 3.9
- JAX, Flax, Optax
- NumPy, Pandas, NetworkX, Matplotlib

See `requirements.txt` for pinned versions.

## Related work

This project is part of a broader research program on transport and optimization in physical networks:

- **G. Gounaris, E. Katifori.** *Braess's Paradox Analog in Physical Networks of Optimal Exploration.* Phys. Rev. Lett. 133, 067401 (2024). [arXiv:2303.02146](https://arxiv.org/abs/2303.02146)
- **G. Gounaris, M. Ruiz Garcia, E. Katifori.** *The Central Role of Metabolism in Vascular Morphogenesis.* [arXiv:2111.04657](https://arxiv.org/abs/2111.04657)

The microvascular network data used in this project is from:

- A.R. Pries, T.W. Secomb, P. Gaehtgens, and J.F. Gross. *Blood flow in microvascular networks — Experiments and simulation.* Circulation Research 67: 826–834 (1990).


## License

MIT
