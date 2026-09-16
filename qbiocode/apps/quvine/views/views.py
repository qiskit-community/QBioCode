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

import networkx as nx


import numpy as np

def constrained_view(
    G,
    root,
    num_views,
    max_nodes,
    max_edges,
    max_degree,
    rng,
    degree_norm=False,
    degree_alpha=0.5,
    eps=1e-6,
):
    """
    Build constrained ego-views with optional degree-normalized expansion.

    Parameters
    ----------
    degree_norm : bool
        If True, neighbors are sampled with probability proportional to
        (deg + eps)^(-degree_alpha)
    degree_alpha : float
        Strength of degree downweighting (0.5 recommended)
    """

    neighbors = list(G.neighbors(root))

    # Isolated root: no edges to traverse, no valid view possible.
    # This can happen if the full graph G has isolated nodes (e.g. last-resort
    # random fill in subgraph.py). Return no views so the walk loop skips it.
    if not neighbors:
        return []

    views = []

    def sample_frontier(nodes):
        if not degree_norm or len(nodes) == 0:
            rng.shuffle(nodes)
            return nodes

        degrees = np.array([G.degree(v) for v in nodes], dtype=float)
        weights = 1.0 / np.power(degrees + eps, degree_alpha)
        weights /= weights.sum()

        # sample without replacement
        return list(rng.choice(nodes, size=len(nodes), replace=False, p=weights))

    for _ in range(num_views):
        view_nodes = {root}
        view_edges = 0

        frontier = sample_frontier(neighbors.copy())

        while frontier:
            candidate = frontier.pop()

            if candidate in view_nodes:
                continue

            # count incremental edges only
            new_edges = sum(
                1 for v in view_nodes if G.has_edge(candidate, v)
            )

            if view_edges + new_edges > max_edges:
                continue

            if new_edges > max_degree:
                continue

            view_nodes.add(candidate)
            view_edges += new_edges

            # expand frontier
            for nbr in G.neighbors(candidate):
                if nbr in view_nodes:
                    continue

                if not degree_norm:
                    frontier.append(nbr)
                else:
                    # probabilistic degree-normalized frontier growth
                    deg = G.degree(nbr)
                    p_keep = 1.0 / np.power(deg + eps, degree_alpha)
                    if rng.random() < min(1.0, p_keep):
                        frontier.append(nbr)

            if len(view_nodes) >= max_nodes or view_edges >= max_edges:
                break

        # Only keep views where root has at least one neighbour (len > 1).
        # A {root}-only view can arise when all frontier candidates fail the
        # constraint checks; passing it to a quantum walk causes a crash.
        if len(view_nodes) > 1:
            views.append(view_nodes)

    return views


# def constrained_view(G, 
#                     root, 
#                     num_views, 
#                     max_nodes,
#                     max_edges,
#                     max_degree,
#                     rng
#                     ): 
#     neighbors = list(G.neighbors(root))
#     views = [] 
#     for _ in range(num_views):
#         view_nodes = {root}
#         view_edges = 0 
        
#         #randomize expansion order 
#         frontier = list(neighbors)
#         rng.shuffle(frontier)
                
#         while frontier: 
#             candidate_node = frontier.pop()
            
#             if candidate_node in view_nodes:
#                 continue 
            
#             #compute incremental edges only 
#             new_edges = sum(1 for v in view_nodes if G.has_edge(candidate_node, v))
            
#             if view_edges + new_edges > max_edges: 
#                 continue 
            
#             #local degree check 
#             if new_edges > max_degree: 
#                 continue 
            
#             view_nodes.add(candidate_node) 
#             view_edges += new_edges 
            
#             for neighbor in G.neighbors(candidate_node): 
#                 if neighbor not in view_nodes: 
#                     frontier.append(neighbor)
                
#             #enforce sizes 
#             if len(view_nodes) >= max_nodes or view_edges >= max_edges: 
#                 break 
            
#         views.append(view_nodes)
        
#     return views 

