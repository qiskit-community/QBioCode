# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SGNS embedding core: views -> walks -> corpus -> word2vec.

These functions are extracted verbatim from ``Pipeline._run_single_iteration``
and its private helpers so that both the Hydra pipeline and the
:func:`quvine.embed` API share a single, behavior-preserving code path.

Reproducibility invariants that MUST be preserved (the cluster pipeline depends
on them):

* Per-root RNG seed: ``cfg.experiment.base_seed + 10000 * it + idx`` where
  ``idx`` indexes ``sorted(roots)``.
* Embedding row order is ``list(graph.nodes)`` (never ``sorted``).
* ``train_embeddings`` reads the top-level ``cfg.min_count``.
* joblib ``Parallel(backend='loky', batch_size=1, prefer='processes')``.

Note on the ``kinds`` argument: when a subset of walk kinds is requested, only
those walkers run, which changes the per-root RNG stream relative to a full
multi-kind run. Calling ``run_sgns`` with ``kinds=list(cfg.walks.kinds)`` (as
the pipeline does) reproduces the original behavior exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from qbiocode.apps.quvine.corpus.builder import CorpusBuilder
from qbiocode.apps.quvine.embedding.word2vec import corpus_to_embedding
from qbiocode.apps.quvine.views.generator import ViewBuilder
from qbiocode.apps.quvine.walks.base import BaseWalker

# Below this many roots a serial loop is used (matches original pipeline).
_PARALLEL_ROOT_THRESHOLD = 2000


def chunkify(seq, chunk_size):
    """Yield successive ``chunk_size``-sized chunks of ``seq``."""
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


def build_views(cfg, graph, root, rng):
    """Build constrained views for a single root."""
    return ViewBuilder(cfg=cfg, rng=rng).build(graph, root)


def run_walks_for_root(cfg, graph, root, views, rng):
    """Run every configured walk kind over each view for a single root."""
    walker = BaseWalker(cfg=cfg, rng=rng)
    all_walks = {k: [] for k in cfg.walks.kinds}

    for view in views:
        view_g = graph.subgraph(view)
        view_nodes = list(view_g.nodes())
        out = walker.run(graph, root, view_nodes)
        for walk_kind, walks in out.items():
            all_walks[walk_kind].extend(walks)

    return all_walks


def process_root(cfg, graph, root, node2idx, it):
    """Build views + walks for a single root with deterministic seeding."""
    idx = node2idx[root]
    seed = (cfg.experiment.base_seed + 10000 * it + idx)
    rng = np.random.default_rng(seed)

    views = build_views(cfg, graph, root, rng)
    walk_outputs = run_walks_for_root(cfg, graph, root, views, rng)

    if not walk_outputs or all(len(walks) == 0 for walks in walk_outputs.values()):
        return root, {}
    return root, walk_outputs


def process_root_chunk(cfg, graph, roots, node2idx, it):
    """Process a batch of roots inside a single worker process."""
    results = []
    for root in roots:
        root, walk_outputs = process_root(cfg, graph, root, node2idx, it)
        results.append((root, walk_outputs))
    return results


def build_corpora(
    cfg,
    graph: nx.Graph,
    it: int = 0,
    *,
    n_jobs: int = 1,
    chunk_size: int = 30,
) -> Dict[str, List[List[str]]]:
    """
    Run walks over every root and compile a per-walk-kind token corpus.

    Returns a dict ``{walk_kind: corpus}`` where each corpus is a flat list of
    walks (each walk a list of node-id strings).
    """
    roots = list(graph.nodes)
    node2idx = {node: i for i, node in enumerate(sorted(roots))}
    corpus_builder = {kind: CorpusBuilder() for kind in cfg.walks.kinds}

    n_roots = len(roots)
    if n_roots < _PARALLEL_ROOT_THRESHOLD or n_jobs == 1:
        chunks = [roots]
        effective_jobs = 1
    else:
        chunks = list(chunkify(roots, chunk_size))
        effective_jobs = n_jobs

    parallel = Parallel(
        n_jobs=effective_jobs,
        backend="loky",
        batch_size=1,
        prefer="processes",
    )

    valid_roots = 0
    for chunk_results in parallel(
        delayed(process_root_chunk)(cfg, graph, chunk, node2idx, it)
        for chunk in chunks
    ):
        for root, walk_outputs in chunk_results:
            if not walk_outputs or all(len(w) == 0 for w in walk_outputs.values()):
                continue
            valid_roots += 1
            for walk_kind, walks in walk_outputs.items():
                if len(walks) == 0:
                    continue
                corpus_builder[walk_kind].add(root, walks)

    assert valid_roots > 0, "No valid roots with walks were found."

    return {kind: builder.build() for kind, builder in corpus_builder.items()}


def train_embeddings(cfg, graph: nx.Graph, all_corpora: Dict[str, List[List[str]]]):
    """Train one SGNS (word2vec) embedding per walk kind. Rows in node order."""
    embeddings = {}
    for kind, corpus in all_corpora.items():
        embeddings[kind] = corpus_to_embedding(
            corpus=corpus,
            nodes=graph.nodes,
            vector_size=cfg.train.embedding_dim,
            window=cfg.train.window,
            sg=cfg.train.sg,
            negative=cfg.train.negative,
            min_count=cfg.min_count,
            workers=cfg.train.workers,
            epochs=cfg.train.epochs,
        )
    return embeddings


def run_sgns(
    cfg,
    graph: nx.Graph,
    it: int = 0,
    *,
    kinds: Optional[List[str]] = None,
    n_jobs: int = 1,
    chunk_size: int = 30,
) -> Dict[str, np.ndarray]:
    """
    One-shot SGNS embedding: ``build_corpora`` -> ``train_embeddings``.

    Args:
        cfg: OmegaConf config (reads ``walks.*``, ``views.*``, ``train.*``,
            ``min_count``, ``experiment.base_seed``).
        graph: NetworkX graph.
        it: Iteration index used in the per-root seed.
        kinds: Walk kinds to compute. ``None`` uses ``cfg.walks.kinds``.
        n_jobs, chunk_size: joblib parallelism over roots.

    Returns:
        ``{walk_kind: embedding}`` with each embedding ``(n_nodes, dim)`` in
        ``list(graph.nodes)`` order.
    """
    if kinds is not None:
        cfg = OmegaConf.merge(cfg, {"walks": {"kinds": list(kinds)}})

    all_corpora = build_corpora(
        cfg, graph, it, n_jobs=n_jobs, chunk_size=chunk_size
    )
    return train_embeddings(cfg, graph, all_corpora)
