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

import os
import pandas as pd 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import random
import warnings

def get_stats(graph): 
    
    stats = {}
    stats['num_nodes'] = graph.number_of_nodes() 
    stats['num_edges'] = graph.number_of_edges()
    stats['density'] = nx.density(graph)
    degrees = [d for n,d in graph.degree()]
    stats['average_degree'] = np.round(np.mean(degrees), decimals=3)
    stats['average_clustering_coefficient'] = np.round(nx.average_clustering(graph), decimals=3)
    betweenness = list(nx.betweenness_centrality(graph).values())
    stats['average_betweenness'] = np.round(np.mean(betweenness), decimals=3)
    stats['average_assortavity_coefficient'] = np.round(nx.degree_assortativity_coefficient(graph),
                                                        decimals=3)
    stats['num_connected_components'] = nx.number_connected_components(graph)
    largest_cc = max(nx.connected_components(graph), key=len)
    stats['largest_cc_size'] = len(largest_cc)
    
    return stats 
    
def draw_graph(cfg, G, source=None, target=None, title="Graph", file_path=None):
    fig = plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    default_color = "#185e8d"
    d = dict(G.degree())
    node_colors = [default_color] * len(G.nodes)
    node_to_index = {node:i for i, node in enumerate(G.nodes)}
    if source is None and target is None:
        nx.draw_networkx(G, 
                pos = pos, 
                with_labels=True,
                nodelist=list(d.keys()), 
                node_color=node_colors,
                node_size=[v*6 for v in d.values()],
                width=2)
    else: 
        for s,t in zip(source, target):
            for i, node in enumerate([s,t]): 
                if node in node_to_index:
                    if i == 0:
                        node_colors[node_to_index[node]] = 'red'
                    elif i == 1: 
                        node_colors[node_to_index[node]] = 'green'
                else:
                    if cfg.draw.verbose:
                        print(f"Node {node} not found in graph")
        nx.draw_networkx(G, 
                pos = pos, 
                with_labels=False,
                nodelist=list(d.keys()), 
                node_color=node_colors,
                node_size=[v*6 for v in d.values()],
                width=2)
    plt.title(title)
    if file_path is not None: 
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        
def sample_walks_from_distribution(
    rwr_scores,
    num_walks=10,
    walk_length=40,
    rng=None,
    seed=None
):
    """
    Generate pseudo-walks by sampling from RWR stationary distribution.

    Parameters
    ----------
    rwr_scores : dict
        Node -> probability
    num_walks : int
    walk_length : int
    seed : int or None

    Returns
    -------
    List[List[node]]
    """
    
    if rng is None and seed is not None:
        warnings.warn(
            "Using 'seed' is deprecated. Pass a numpy.random.Generator instead.",
            DeprecationWarning,
        )
    
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        use_rng = rng
        use_numpy = True
    else:
        # legacy behavior
        use_rng = random.Random(seed)
        use_numpy = False

    nodes = list(rwr_scores.keys())
    probs = np.array([rwr_scores[n] for n in nodes], dtype=float)
    probs /= probs.sum()

    walks = []
    for _ in range(num_walks):
        if use_numpy:
            walk = use_rng.choice(
                nodes,
                size=walk_length,
                replace=True,
                p=probs,
            ).tolist()
        else:
            walk = use_rng.choices(
                nodes,
                weights=probs.tolist(),
                k=walk_length,
            )
        walks.append(walk)
    
    return walks

def plot_metric(cfg, df, metric, file_path=None):
    
    methods = sorted(df["method"].unique())

    for k in cfg.eval.k_values:
        subset = df[(df.metric == metric) & (df.k == k)]

        agg = (
            subset
            .groupby(["method", "control"], as_index=False)
            .agg(
                mean=("mean", "mean"),
                std=("mean", "std"),
            )
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        x = np.arange(len(methods))   # ← FIX
        width = 0.25

        for i, control in enumerate(
            ["true", "degree_matched", "distance_matched"]
        ):
            vals = (
                agg[agg.control == control]
                .set_index("method")
                .reindex(methods)
            )

            ax.bar(
                x + i * width,
                vals["mean"].values,
                width,
                yerr=vals["std"].values,
                label=control.replace("_", " ").title(),
                capsize=3,
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel(f"{metric.title()}@{k}")
        ax.set_title(f"{metric.title()}@{k} (Seed→Target Prioritization)")
        ax.legend()
        plt.tight_layout()        
        if file_path is not None: 
            out_fname = metric+'@'+str(k)+'_ranking.png' 
            out_fname = os.path.join(file_path, out_fname)
            fig.savefig(out_fname, dpi=300, bbox_inches='tight')
    
    
def plot_precision_recall(df, control="true", file_path=None):
    
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = list(set(df['method']))
    for method in methods:
        rec = df[
            (df.method == method) &
            (df.control == control) &
            (df.metric == "recall")
        ].sort_values("k")

        prec = df[
            (df.method == method) &
            (df.control == control) &
            (df.metric == "precision")
        ].sort_values("k")

        ax.plot(
            rec["mean"],
            prec["mean"],
            marker="o",
            label=method
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall ({control.replace('_',' ')})")
    ax.legend()
    plt.tight_layout()
    
    if file_path is not None: 
        out_fname = 'Precision_Recall_'+control+'.png' 
        out_fname = os.path.join(file_path, out_fname)
        fig.savefig(out_fname, dpi=300, bbox_inches='tight')
        
def plot_metric_vs_k(df, metric='precision', control="true", file_path=None):

    fig = plt.figure(figsize=(6,4))
    
    for method in df.method.unique():
        subset = df[
            (df.method == method) &
            (df.control == control) &
            (df.metric == metric)
        ].sort_values("k")

        plt.plot(
            subset["k"],
            subset["mean"],
            marker="o",
            label=method
        )

    plt.xlabel("K")
    plt.ylabel(metric + "@K")
    plt.title(metric + " vs K (Seed→Target Prioritization)")
    plt.legend()
    plt.tight_layout()
    
    if file_path is not None: 
        out_fname = metric+'vs_k_'+control+'.png'
        out_fname = os.path.join(file_path, out_fname)
        fig.savefig(out_fname, dpi=300, bbox_inches='tight')
