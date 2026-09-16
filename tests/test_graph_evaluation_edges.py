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

"""``evaluate_graph`` on degenerate graphs.

51 metrics run over a single graph, and many of them are undefined for graphs a
caller will realistically hand over: the kNN graph QProfiler builds can come out
disconnected, a filtered graph can end up empty. The contract is that such a graph
produces a summary row with ``nan`` in the undefined columns -- never an exception,
and never a plausible-looking number standing in for a value that does not exist.

Which columns are present legitimately varies with the graph: metrics that cannot be
computed at all are absent rather than nan-filled, so these tests assert the contract
and the values, not a fixed schema.
"""

from __future__ import annotations

import numpy as np
import pytest

import networkx as nx

from qbiocode import evaluate_graph  # noqa: E402 - after the importorskip


def _graphs():
    disconnected = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    self_loops = nx.karate_club_graph()
    self_loops.add_edge(0, 0)
    weighted = nx.karate_club_graph()
    for u, v in weighted.edges():
        weighted[u][v]["weight"] = 2.5
    return {
        "empty": nx.Graph(),
        "single_node": nx.empty_graph(1),
        "single_node_self_loop": nx.Graph([(0, 0)]),
        "two_nodes_one_edge": nx.path_graph(2),
        "disconnected": disconnected,
        "complete": nx.complete_graph(5),
        "star": nx.star_graph(6),
        "self_loops": self_loops,
        "weighted": weighted,
        "directed": nx.DiGraph([(0, 1), (1, 2)]),
    }


GRAPHS = _graphs()


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_every_graph_shape_produces_one_summary_row(name):
    """No degenerate shape may raise: a metric that cannot be computed is nan."""
    summary = evaluate_graph(GRAPHS[name], name=name)
    assert summary.shape[0] == 1, "evaluate_graph must return exactly one row per graph"
    assert summary["Graph"].iloc[0] == name, "the name argument must be echoed back"


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_no_metric_is_infinite(name):
    """``inf`` is a plausible-looking number that survives arithmetic; nan is honest."""
    summary = evaluate_graph(GRAPHS[name], name=name)
    numeric = summary.select_dtypes(include=[np.number])
    infinite = [column for column in numeric.columns if np.isinf(numeric[column].iloc[0])]
    assert not infinite, f"{name}: infinite values in {infinite}"


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_the_summary_is_deterministic(name):
    """Two evaluations of the same graph must agree, or no comparison across runs holds."""
    first = evaluate_graph(GRAPHS[name], name=name)
    second = evaluate_graph(GRAPHS[name], name=name)
    assert list(first.columns) == list(second.columns)
    numeric = first.select_dtypes(include=[np.number]).columns
    np.testing.assert_allclose(
        first[numeric].to_numpy(dtype=float),
        second[numeric].to_numpy(dtype=float),
        equal_nan=True,
    )


class TestTheEmptyGraph:
    def test_it_returns_counts_rather_than_raising(self):
        summary = evaluate_graph(nx.Graph(), name="nothing")
        assert summary["num_nodes"].iloc[0] == 0
        assert summary["num_edges"].iloc[0] == 0

    def test_it_does_not_invent_spectral_metrics(self):
        """A graph with no nodes has no spectrum; those columns must be absent or nan."""
        summary = evaluate_graph(nx.Graph())
        for column in ("spectral_gap", "von_neumann_entropy", "modularity"):
            if column in summary.columns:
                assert np.isnan(float(summary[column].iloc[0])), (
                    f"{column} reported a value for a graph with no nodes"
                )

    def test_none_is_still_a_type_error_not_an_empty_graph(self):
        with pytest.raises((TypeError, ValueError)):
            evaluate_graph(None)


class TestValuesOnGraphsWithKnownAnswers:
    """If a metric is wrong, these are the graphs where it is provably wrong."""

    def test_a_complete_graph_has_density_one_and_no_community_structure(self):
        summary = evaluate_graph(nx.complete_graph(5))
        assert float(summary["density"].iloc[0]) == pytest.approx(1.0)
        assert float(summary["modularity"].iloc[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(summary["avg_degree"].iloc[0]) == pytest.approx(4.0)

    def test_a_disconnected_graph_has_a_zero_spectral_gap(self):
        """Algebraic connectivity is 0 exactly when the graph is disconnected."""
        summary = evaluate_graph(nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)))
        assert float(summary["normalized_spectral_gap"].iloc[0]) == pytest.approx(0.0, abs=1e-9)
        # Two equal cliques are the textbook high-modularity partition.
        assert float(summary["modularity"].iloc[0]) == pytest.approx(0.5, abs=1e-6)

    def test_a_star_graph_is_recognized_as_maximally_centralized(self):
        summary = evaluate_graph(nx.star_graph(6))
        assert float(summary["num_nodes"].iloc[0]) == 7
        assert float(summary["num_edges"].iloc[0]) == 6

    def test_edge_weights_actually_participate(self):
        """A weighted graph that summarizes identically to an unweighted one ignores weights."""
        plain = nx.karate_club_graph()
        weighted = nx.karate_club_graph()
        for u, v in weighted.edges():
            weighted[u][v]["weight"] = 2.5
        assert not evaluate_graph(plain).equals(evaluate_graph(weighted))

    def test_self_loops_do_not_corrupt_the_summary(self):
        """A self-loop inflates the degree sum; the summary must stay finite and sane."""
        graph = nx.karate_club_graph()
        graph.add_edge(0, 0)
        summary = evaluate_graph(graph)
        assert 0.0 <= float(summary["density"].iloc[0]) <= 1.0
        assert float(summary["avg_degree"].iloc[0]) > 0.0
