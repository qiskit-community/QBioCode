# Random Graph Generators for QuVINE

This module provides comprehensive random graph generation capabilities for testing and evaluating embedding algorithms in QuVINE.

## Quick Start

```python
from quvine.data import generate_barabasi_albert, get_graph_statistics

# Generate a scale-free network
G = generate_barabasi_albert(n=100, m=3, seed=42)

# Get statistics
stats = get_graph_statistics(G)
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
```

## Available Generators

### Classical Models
- **Erdős-Rényi**: Random graphs with uniform edge probability
- **Barabási-Albert**: Scale-free networks (power-law degree distribution)
- **Watts-Strogatz**: Small-world networks (high clustering, short paths)
- **Powerlaw Cluster**: Power-law degree with high clustering

### Structured Models
- **Stochastic Block Model**: Custom community structure
- **Modular Network**: Simplified community generation
- **Hierarchical Network**: Tree-like structures with random edges
- **Core-Periphery**: Dense core with sparse periphery
- **Bipartite**: Two-mode networks
- **Random Geometric**: Spatial networks

### Special Features
- **Add Hub Nodes**: Augment graphs with highly connected nodes
- **Seeds & Targets**: Generate graphs with labeled nodes for evaluation
- **Graph Statistics**: Comprehensive metrics computation

## Documentation

- **Full Guide**: `docs/random_graphs_guide.md`
- **Quick Reference**: `docs/random_graphs_quick_reference.md`
- **Examples**: `examples/random_graph_examples.py`
- **Tests**: `tests/test_random_graphs.py`

## Use Cases

1. **Testing Embeddings**: Generate controlled graphs to test embedding quality
2. **Benchmarking**: Compare algorithms across different graph structures
3. **Validation**: Verify that embeddings capture specific graph properties
4. **Research**: Study how graph structure affects embedding performance

## Example: Embedding Evaluation

```python
from quvine.data import generate_graph_with_seeds_and_targets

# Generate test graph with known seeds and targets
G, seeds, targets = generate_graph_with_seeds_and_targets(
    n=500,
    num_seeds=20,
    num_targets=30,
    graph_type='modular',
    num_communities=10,
    p_intra=0.3,
    p_inter=0.01,
    seed=42
)

# Use with QuVINE pipeline
# ... run your embedding and evaluation
```

## Integration with QuVINE

The random graph generators are fully integrated with the QuVINE pipeline and can be used with:
- Random walks (RWR, CTQW, DTQW)
- Embedding methods (Word2Vec, Node2Vec)
- Evaluation metrics
- Visualization tools

## Citation

If you use these random graph generators in your research, please cite the QuVINE paper:

```
Quantum-enhanced Network Embeddings via Multi-view Integration for Precision Medicine
A. Bose, F. Utro and L. Parida, 2026. (Under Review)