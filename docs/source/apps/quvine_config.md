# QuVINE Configuration Guide

This guide explains QuVINE's YAML configuration: what each section controls, and
— just as importantly — **which of QuVINE's three entry points actually reads
it**. The shipped default is
[`qbiocode/apps/quvine/configs/config.yaml`](https://github.com/qiskit-community/QBioCode/blob/main/qbiocode/apps/quvine/configs/config.yaml).

## Overview

One file serves three consumers, and they read different parts of it:

| Entry point | How to invoke | Reads |
|---|---|---|
| `embed()` API | `from qbiocode.apps.quvine import embed` | `views`, `walks`, `train`, `min_count`, `fusion`, `experiment.base_seed` |
| `quvine` CLI | `quvine --edgelist … --output …` | the same keys — it calls `embed()` |
| Hydra pipeline | `python -m qbiocode.apps.quvine.main` | **everything**, including `data_path`, `graph`, `disease`, `runtime`, `preprocess`, `baselines`, `analysis`, `evaluation` |

```{important}
The two things people trip over most:

1. **`data_path`, `graph.*` and `disease.*` are read only by the Hydra research
   pipeline.** The `embed()` API and the `quvine` CLI take their graph from the
   caller — a `networkx.Graph` or an edge-list file — and ignore those sections
   entirely. You do not need to set them to embed a graph.
2. **`runtime.run_name` uses `${now:…}`, which is a Hydra resolver.** OmegaConf
   resolves interpolations lazily, so loading this file outside Hydra is fine as
   long as nothing asks for that key. `embed()` never does.
```

An installed copy of QuVINE finds this file automatically. Override the choice
with, in order of precedence: the `config=` argument to `embed()`, the CLI's
`--config`, or the `QUVINE_DEFAULT_CONFIG` environment variable.

## Quick Start

Everything `embed()` reads, and nothing it does not:

```yaml
# Reproducibility
experiment:
  base_seed: 42

# Multi-view graph construction
views:
  num_views: 8
  max_degree: 9
  max_nodes: 80
  max_edges: 350
  degree_norm: true
  degree_alpha: 0.7

# Walks over each view
walks:
  kinds: [rwr, ctqw, dtqw]
  num_walks: 10
  walk_length: 8
  restart_prob: 0.35     # rwr
  steps: 20              # dtqw
  time: 1.2              # ctqw
  coin: grover           # dtqw
  max_iter: 1000

# Skip-gram negative sampling
min_count: 1
train:
  embedding_dim: 64
  window: 5
  sg: 1
  negative: 10
  epochs: 100
  workers: 8

# Combining per-walk-kind embeddings (used by the *_fused methods)
fusion:
  enabled: true
  method: svd
  k: 10
```

Save that as `my_quvine.yaml` and use it:

```bash
quvine --edgelist edges.csv --method quvine_fused --config my_quvine.yaml --output out/
```

```python
from qbiocode.apps.quvine import embed

result = embed(G, "quvine_fused", config="my_quvine.yaml")
```

---

## Configuration Sections

### Input Data (Hydra pipeline only)

```yaml
data_path: ./data

graph:
  name: my_network
  path: ${data_path}/networks/${graph.name}/edges_list_ncbi.csv

disease:
  name: my_phenotype
```

The pipeline expects four files under `data_path`, named after `graph.name` and
`disease.name`:

```text
${data_path}/networks/<graph.name>/edges_list_ncbi.csv
${data_path}/gwas_gene_pvals/<disease.name>/filtered_ncbi_PEGASUS_<disease.name>_gwas_data.csv
${data_path}/gene_seeds/<disease.name>_ncbi_seeds.json
${data_path}/gwas_catalog_targets/<disease.name>_targets_gene2ncbi.json
```

```{note}
Every path in the shipped config is **relative**, and read from the directory you
launch from. Earlier releases named an absolute `/dccstor/…` research filesystem
here, which no installed copy could resolve. `graph.name` and `disease.name` are
neutral placeholders (`my_network`, `my_phenotype`) — set them to your own.
```

### Runtime and Output

```yaml
runtime:
  name: qune
  run_name: ${now:%Y-%m-%d_%H-%M-%S}
  run_dir: ${graph.name}
  output_dir: ./outputs/${runtime.run_dir}/${runtime.run_name}
  n_jobs: 16
  chunk_size: 30

hydra:
  run:
    dir: ${runtime.output_dir}
  sweep:
    dir: ${runtime.output_dir}
```

| Key | Meaning |
|---|---|
| `n_jobs` | joblib workers for the per-root walk loop. `embed()` takes this as its own `n_jobs` argument instead. |
| `chunk_size` | Roots per joblib batch. |
| `output_dir` | Where the pipeline writes results; also what Hydra uses as its run directory. |

```{tip}
Parallelism only engages above a threshold: the walk loop runs serially for
graphs under 2,000 nodes, where joblib's process startup costs more than it
saves. Raising `n_jobs` on a small graph changes nothing.
```

### Reproducibility

```yaml
experiment:
  iterations: 2      # stochastic repetitions; iteration i uses base_seed + i
  base_seed: 42

seed: 42             # used before the iteration loop
```

`base_seed` is the one that matters for `embed()`, and `--base-seed` on the CLI
overrides it. Seeding is per-root and derived, not global: root *idx* in
iteration *it* uses `base_seed + 10000*it + idx`, over the roots in sorted order.
That is what makes a run reproducible independently of how the work was
scheduled across workers.

### Preprocessing (Hydra pipeline only)

```yaml
preprocess:
  subsample:
    enabled: true
    max_nodes: 400
    radius: 4

  sparsify:
    enabled: true
    retain_ratio: 0.4
    max_degree: 40
    scoring: common_neighbors
```

`subsample` takes a radius-bounded neighbourhood around the seed set, capped at
`max_nodes`. `sparsify` drops edges by a local score, keeping `retain_ratio` of
them and capping degree at `max_degree`.

```{warning}
**Migration note.** The pipeline reads
`preprocess.subsample.{enabled,max_nodes,radius}`. An older schema spelled this
`preprocess.subgraph.{num_nodes,max_hops}` — those keys are **not read**, and a
config still using them crashed the run. The comment recording this is kept in
the shipped file.
```

### Views

```yaml
views:
  num_views: 8
  constrained: true
  max_degree: 9
  max_nodes: 80
  max_edges: 350
  degree_norm: true
  degree_alpha: 0.7
```

QuVINE builds `num_views` constrained subgraphs around each root rather than
walking the whole graph. The three caps bound each view's size; `degree_norm`
with `degree_alpha` down-weights hubs when sampling, so a view is not dominated
by whichever high-degree node it happened to touch.

Larger `num_views` gives a richer corpus at linear cost in walk time.

### Walks

```yaml
walks:
  kinds: [rwr, ctqw, dtqw]
  num_walks: 10
  walk_length: 8
  restart_prob: 0.35
  steps: 20
  time: 1.2
  coin: grover
  max_iter: 1000
```

| Key | Applies to | Meaning |
|---|---|---|
| `kinds` | all | Which walks to run. `rwr` is classical; `ctqw` and `dtqw` are quantum. |
| `num_walks`, `walk_length` | all | Walks started per root, and tokens per walk. |
| `restart_prob` | `rwr` | Restart probability of the random walk with restart. |
| `time` | `ctqw` | Evolution time of the continuous-time quantum walk. |
| `steps`, `coin` | `dtqw` | Steps and coin operator of the discrete-time quantum walk. |
| `max_iter` | `ctqw`/`dtqw` | Iteration cap in the walk solver. |

```{tip}
Small step counts often work best. High `steps` or large `time` over-mixes the
walk toward its stationary distribution, which washes out exactly the local
structure the embedding is meant to capture — so a modest value tends to give
comparable or better embeddings, faster.
```

Each entry in `kinds` produces its own embedding. The `*_fused` methods then
combine them; the single-kind methods (`quvine_rwr`, `quvine_ctqw`,
`quvine_dtqw`) use one.

### Training (SGNS)

```yaml
dimension: 64
window: 5
min_count: 1
workers: 8

train:
  embedding_dim: ${dimension}
  window: ${window}
  sg: 1
  negative: 10
  workers: ${workers}
  min_count: ${min_count}
  epochs: 100
```

The four top-level scalars exist so one edit changes every place they are used —
`train`, and the `baselines` below, interpolate them. Change `dimension: 64` and
node2vec's `dimensions` follows.

| Key | Meaning |
|---|---|
| `embedding_dim` | Per-walk-kind embedding width. It is the returned `dim` for single-kind methods; a `*_fused` method returns `fusion.k` instead, because fusion reduces the stacked embeddings to its own rank. |
| `window` | Skip-gram context window over the walk token sequence. |
| `sg` | `1` for skip-gram, `0` for CBOW. |
| `negative` | Negative samples per positive pair. |
| `min_count` | Minimum token frequency. Keep at `1` — a node dropped here is a node with no embedding row. |
| `epochs` | Word2Vec training epochs. |

### Baselines (Hydra pipeline only)

```yaml
baselines:
  node2vec:
    enabled: true
    dimensions: ${dimension}
    walk_length: 8
    num_walks: 10
    p: 1
    q: 0.5
    seed: ${seed}

  gat_ctqw_heat:
    enabled: false
    variant: heat_qcal_ctqw
    embedding_dim: ${dimension}
    heat_t: 1.0
  # ... 9 more gat_*, 10 graphgps_*
```

Twenty-one baselines ship configured and all but `node2vec` are `enabled: false`,
so a default pipeline run is fast. Flip `enabled` to add one to the comparison.

Each `gat_*` / `graphgps_*` entry is the same architecture under a different
spectral filter, named by `variant`:

| `variant` | Filter |
|---|---|
| `raw` | No spectral filter — the plain architecture. |
| `heat_fixed`, `poly_fixed` | Fixed heat-kernel / polynomial filter. |
| `rwr` | Random-walk-with-restart diffusion. |
| `heat_qcal_<walk>`, `poly_qcal_<walk>` | Quantum-**cal**ibrated: the filter is fitted against a `ctqw`, `dtqw` or `rwr` walk. |

The `_qcal_` variants are the interesting ones — they are how a classical
architecture is given a quantum walk's spectral profile, which makes a
like-for-like comparison possible. They need quantum targets, built from the
graph automatically (`baselines.quantum_target_max_nodes`, default 64, bounds
their support).

```{note}
`gat_*` and `graphgps_*` need only the base install. `node2vec` and the walk
methods need `pip install "qbiocode[quvine]"`.
```

### Fusion

```yaml
fusion:
  enabled: true
  method: svd
  k: 10
```

How the per-walk-kind embeddings are combined into one matrix, for the `*_fused`
methods.

| `method` | Approach |
|---|---|
| `svd` | Truncated SVD of the stacked embeddings. **Default; works out of the box.** |
| `graphreg` | Graph-regularized fusion. |
| `attention` | Learned attention weights across views. |
| `hybrid` | Combination of the above. |
| `svd_shared_priv` | SVD split into shared and per-view private subspaces. |
| `all` | Runs every method, for comparison. |

`k` is the fused rank.

```{warning}
`concatenate` is **not** a supported value, despite appearing in some older
configs. The supported set is exactly the six above. Note also that `graphreg`
builds a dense normalized Laplacian, so it is memory-bound on large graphs.
```

### Analysis and Evaluation (Hydra pipeline only)

```yaml
analysis:
  cca_components: 10
  knn_k: 5

evaluation:
  enabled: false
  k_values: [20, 40, 80]
  n_repeats: 20
  deg_tol: 0.1
  centroid: false
  max_seed: true
```

`analysis` configures the embedding comparison (canonical correlation between
embeddings, and k-NN agreement). `evaluation` configures seed-target ranking:
recall@k and precision@k at each `k_values`, against degree-matched null
controls resampled `n_repeats` times within `deg_tol`.

### Output Control

```yaml
save_embeddings: true
compare_embeddings: true
verbose: true
plots: true

draw:
  graph: true
  verbose: false
```

---

## Overriding Without Editing the File

**Python** — a dict or a Hydra-style dotlist, deep-merged over the base:

```python
result = embed(G, "quvine_fused", overrides={"train": {"epochs": 20}})
result = embed(G, "quvine_fused", overrides=["train.epochs=20", "walks.kinds=[rwr]"])
```

**Hydra pipeline** — dotlist arguments on the command line:

```bash
python -m qbiocode.apps.quvine.main train.epochs=20 experiment.iterations=5
```

**`quvine` CLI** — `--config` for a whole file, `--base-seed` for the one key
worth overriding on its own:

```bash
quvine --edgelist edges.csv --config my_quvine.yaml --base-seed 7 --output out/
```

---

## Best Practices

```{tip}
1. **Start from the shipped config** rather than a blank file — the interpolations
   between `dimension`/`window`/`workers` and `train`/`baselines` are easy to
   break by hand.
2. **Set `experiment.base_seed` explicitly** in anything you intend to compare
   across runs.
3. **Keep `walks.steps` and `walks.time` modest.** Over-mixed quantum walks cost
   more and embed worse.
4. **Change `dimension`, not `train.embedding_dim`** — the former propagates to
   the baselines, so the comparison stays like-for-like.
5. **Raise `views.num_views` before `walks.num_walks`** when the corpus is too
   small; view diversity buys more than more walks over the same views.
```

---

## Troubleshooting

**"Could not locate a default QuVINE config.yaml"**
- Pass `config=<path>` to `embed()`, or `--config` to the CLI, or set
  `QUVINE_DEFAULT_CONFIG`.

**A relative `data_path` resolves to the wrong place**
- Relative paths are read from your *working directory*, not from the config
  file's location. Launch from the project root, or use an absolute path in your
  own config.

**`InterpolationResolutionError: unsupported interpolation "now"`**
- Something asked for `runtime.run_name` outside Hydra. `${now:…}` is a Hydra
  resolver. Either run through `main.py`, or set `run_name` to a literal string
  in your own config.

**Config keys I set have no effect**
- Check the table at the top: `embed()` and the CLI read only `views`, `walks`,
  `train`, `min_count`, `fusion` and `experiment.base_seed`. Everything else is
  the Hydra pipeline's.

**`preprocess.subgraph` is ignored**
- That schema was replaced. Use
  `preprocess.subsample.{enabled,max_nodes,radius}`.

**A method is missing from `list_methods()`**
- Its dependency is absent. Inspect
  `qbiocode.apps.quvine.baselines.UNAVAILABLE`, which maps each unimportable
  baseline module to the reason, or run
  `print(qbiocode.apps.quvine.describe_environment())`.

---

## See Also

- {doc}`QuVINE Usage Guide <quvine>` — how to run QuVINE, and the method list
- {doc}`QProfiler Configuration <config>` — the analogous guide for QProfiler
- {doc}`Tutorial Notebooks <../tutorials>` — worked examples
