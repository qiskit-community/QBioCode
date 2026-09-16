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
Command-line interface for the QuVINE embedding app.

Embeds the nodes of a graph (given as an edge list) with a chosen QuVINE method
and writes the embedding matrix + metadata. Mirrors the qsage/qprofiler CLIs.

Usage:
    quvine --edgelist edges.csv --method quvine_fused --output out/
    quvine --edgelist edges.tsv --sep '\\t' --method node2vec --output out/
    quvine --list-methods
"""

import argparse
import json
import os
import sys


#: Conventional first-row labels of an edge-list file. A row whose first two
#: fields are *both* in this set is a header, not an edge.
#:
#: Matched against a closed set of names rather than sniffed. The obvious
#: heuristics all have a failure mode on real data -- "row 0's endpoints appear
#: nowhere else" misreads a sparse matching's first edge, and "row 0 is
#: non-numeric" misreads every string-labelled graph -- and misreading a real
#: edge as a header silently deletes it. Single letters (``u``, ``v``, ``a``,
#: ``b``) are deliberately absent: they are plausible node names. Anything this
#: set does not cover is handled by ``--header yes``.
_HEADER_NAMES = frozenset(
    {
        "source", "target", "from", "to", "src", "dst", "head", "tail",
        "node1", "node2", "node_1", "node_2", "node_a", "node_b",
        "source_id", "target_id", "sourceid", "targetid",
        "gene1", "gene2", "gene_1", "gene_2", "gene_a", "gene_b",
        "protein1", "protein2", "protein_1", "protein_2",
    }
)


def _looks_like_a_header(row) -> bool:
    """True when this row is column names rather than an edge."""
    try:
        first, second = str(row.iloc[0]).strip().lower(), str(row.iloc[1]).strip().lower()
    except (IndexError, AttributeError):
        return False
    return first in _HEADER_NAMES and second in _HEADER_NAMES


def _load_graph(path: str, sep: str, weighted: bool, header: str = "auto"):
    """Load an undirected graph from a 2- or 3-column edge list (CSV/TSV).

    ``header`` is ``"auto"``, ``"yes"`` or ``"no"``.

    A header row is dropped rather than read as an edge. It used to be read as
    one: ``pd.read_csv(..., header=None)`` turned the ``source,target`` line that
    this tool's own ``--help`` and documentation show into an edge between a node
    called "source" and a node called "target". Nothing failed -- the run
    produced an embedding with two extra rows and one edge that is not in the
    graph, which is the kind of wrong that survives review.
    """
    import pandas as pd
    import networkx as nx

    try:
        df = pd.read_csv(path, sep=sep, engine="python", header=None, comment="#")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        # Narrow to the four ways a text table can fail to parse, so a genuine bug
        # in this function still raises rather than being reported as a bad file.
        raise ValueError(
            f"Could not read the edge list {path!r} as a {sep!r}-separated table "
            f"({type(exc).__name__}: {exc}). Check --sep -- use --sep '\\t' for TSV."
        ) from exc
    if df.shape[1] < 2:
        raise ValueError(
            f"Edge list {path!r} must have at least 2 columns (source, target); "
            f"found {df.shape[1]}."
        )

    if header == "yes" or (header == "auto" and len(df) and _looks_like_a_header(df.iloc[0])):
        if not len(df):
            raise ValueError(
                f"Edge list {path!r} was read as header-only (--header yes) and "
                f"contains no edges."
            )
        dropped = list(df.iloc[0][:2])
        df = df.iloc[1:].reset_index(drop=True)
        print(f"  skipping header row {dropped} in {path}", file=sys.stderr)
        if df.empty:
            raise ValueError(
                f"Edge list {path!r} contains only a header row and no edges."
            )

    df = df.rename(columns={0: "source", 1: "target"})
    # QuVINE's corpus builder requires string node tokens (word2vec); coerce so
    # integer-id edge lists work too.
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    if weighted and df.shape[1] >= 3:
        df = df.rename(columns={2: "weight"})
        return nx.from_pandas_edgelist(df, "source", "target", edge_attr="weight")
    return nx.from_pandas_edgelist(df, "source", "target")


def main():
    """CLI entry point for ``quvine``."""
    # `list_methods` resolves through api.aliases, which is stdlib-only, so
    # --help and --list-methods work without the [quvine] extra installed.
    # `embed` is imported at its point of use further down, because it pulls in
    # omegaconf and the rest of the extra.
    from qbiocode.apps.quvine import list_methods

    parser = argparse.ArgumentParser(
        description="QuVINE: quantum view-based network embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quvine --edgelist edges.csv --method quvine_rwr --output out/
  quvine --edgelist edges.csv --method quvine_fused --output out/
  quvine --list-methods

For more information, see: https://github.com/qiskit-community/QBioCode
""",
    )
    parser.add_argument("--edgelist", "-i", help="Path to a 2/3-column edge-list file (source,target[,weight]).")
    parser.add_argument("--output", "-o", help="Output directory for the embedding + metadata.")
    parser.add_argument(
        "--method", "-m", default="quvine_fused",
        help="Embedding method (default: quvine_fused). See --list-methods.",
    )
    parser.add_argument("--sep", default=",", help="Edge-list column separator (default: ',').")
    parser.add_argument("--weighted", action="store_true", help="Use a 3rd column as edge weight.")
    parser.add_argument(
        "--header", choices=("auto", "yes", "no"), default="auto",
        help="Whether the edge list's first row is column names (default: auto -- "
             "recognized when both of its first two fields are conventional "
             "column names such as source/target).",
    )
    parser.add_argument("--base-seed", type=int, default=None, help="Override the base random seed.")
    parser.add_argument("--config", default=None, help="Path to a QuVINE config YAML (defaults to packaged config).")
    parser.add_argument("--npy", action="store_true", help="Also write the embedding as a .npy array.")
    parser.add_argument("--list-methods", action="store_true", help="List available methods and exit.")
    args = parser.parse_args()

    if args.list_methods:
        for name in list_methods():
            print(name)
        return

    if not args.edgelist or not args.output:
        parser.error("--edgelist/-i and --output/-o are required (unless --list-methods).")
    if not os.path.exists(args.edgelist):
        print(f"Error: edge list {args.edgelist!r} not found.", file=sys.stderr)
        sys.exit(1)
    if os.path.isdir(args.edgelist):
        print(
            f"Error: --edgelist must be a file, but {args.edgelist!r} is a directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.config is not None and not os.path.isfile(args.config):
        # Checked here rather than left to load_config: a missing override path
        # silently fell back to the packaged config in some code paths, so a
        # mistyped --config produced a successful run with the wrong settings.
        print(
            f"Error: --config {args.config!r} is not a file. Omit --config to use "
            f"the packaged default.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.sep:
        print(
            "Error: --sep must be a non-empty separator (e.g. ',' or '\\t').",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate through resolve_method rather than a membership test, so the
    # message carries its close-match suggestions and the accepted set can never
    # drift from what embed() actually dispatches.
    from qbiocode.apps.quvine.api.aliases import resolve_method

    try:
        resolve_method(args.method)
    except KeyError as exc:
        print(
            f"Error: {exc.args[0]} Run 'quvine --list-methods' to see all options.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 72)
    print("QuVINE embedding")
    print("=" * 72)
    print(f"Edge list : {args.edgelist}")
    print(f"Method    : {args.method}")
    print(f"Output    : {args.output}")

    print("\nLoading graph...")
    try:
        G = _load_graph(args.edgelist, args.sep, args.weighted, args.header)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    if G.number_of_nodes() == 0:
        # Every method here embeds nodes; with none there is nothing to write, and
        # the walk builders fail several frames deeper with an empty-corpus error.
        print(
            f"Error: {args.edgelist!r} produced a graph with no nodes. An edge "
            f"list needs at least one source,target row (lines starting with '#' "
            f"are treated as comments).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Embedding...")
    # A missing [quvine] dependency is a configuration problem, not a crash:
    # print the actionable message and exit 1 instead of dumping a traceback.
    from qbiocode.apps.quvine._deps import QuvineDependencyError

    try:
        from qbiocode.apps.quvine import embed

        result = embed(G, args.method, config=args.config, base_seed=args.base_seed)
    except QuvineDependencyError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  method={result.method} dim={result.dim} time={result.execution_time:.2f}s")

    # Write embedding as node,dim_0,...,dim_{d-1}
    import pandas as pd
    emb_df = pd.DataFrame(result.embedding, index=result.node_order)
    emb_df.columns = [f"dim_{i}" for i in range(result.dim)]
    emb_df.index.name = "node"
    csv_path = os.path.join(args.output, "embedding.csv")
    emb_df.to_csv(csv_path)
    print(f"  wrote {csv_path}")

    if args.npy:
        import numpy as np
        npy_path = os.path.join(args.output, "embedding.npy")
        np.save(npy_path, result.embedding)
        print(f"  wrote {npy_path}")

    meta = {
        "requested_method": result.requested_method,
        "method": result.method,
        "kind": result.kind,
        "dim": result.dim,
        "n_nodes": len(result.node_order),
        "execution_time": result.execution_time,
        "used_quantum_targets": result.used_quantum_targets,
    }
    meta_path = os.path.join(args.output, "embedding_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"  wrote {meta_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
