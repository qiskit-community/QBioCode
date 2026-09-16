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

import random
import networkx as nx 
from qbiocode.apps.quvine.utils.utilities import sample_walks_from_distribution


def get_RWR_pagerank_scores(
    G,
    root,
    restart_prob=0.15,
    view_nodes=None,
    weight=None,
    tol=1e-6,
    max_iter=500
):
    """
    Random Walk with Restart using PageRank.

    Parameters
    ----------
    G : networkx.Graph
    root : node
        Restart node
    restart_prob : float
        Alpha (restart probability)
    view_nodes : set or None
        Optional constraint: restrict graph to these nodes
    weight : str or None
        Edge weight attribute
    tol : float
        Convergence tolerance
    max_iter : int
        Max iterations

    Returns
    -------
    dict
        Node -> RWR stationary probability
    """
    if view_nodes is not None:
        G = G.subgraph(view_nodes)

    if root not in G:
        raise ValueError("Root node not in graph or view")

    # Personalization vector (restart distribution)
    personalization = {n: 0.0 for n in G.nodes()}
    personalization[root] = 1.0

    try:
        pr = nx.pagerank(
            G,
            alpha=1 - restart_prob,
            personalization=personalization,
            weight=weight,
            tol=tol,
            max_iter=max_iter
        )
    except nx.PowerIterationFailedConvergence:
        # fallback
        pr = {v: 1.0 / G.number_of_nodes() for v in G}
        
    return pr


def generate_RWR_pagerank_walks(G, 
                                root, 
                                view_nodes=None, 
                                num_walks=10, 
                                walk_length=6, 
                                restart_prob=0.5,
                                max_iter=100, 
                                rng=None):
    
    rwr_scores = get_RWR_pagerank_scores(
        G,
        root,
        restart_prob=restart_prob,
        view_nodes=view_nodes,
        max_iter=max_iter
    )
    walks = sample_walks_from_distribution(
        rwr_scores,
        num_walks=num_walks,
        walk_length=walk_length, 
        rng=rng
    )
    
    return walks


    
