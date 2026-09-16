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
from qbiocode.apps.quvine._deps import require_module
from qbiocode.apps.quvine.utils.utilities import sample_walks_from_distribution


def _hiperwalk():
    """Resolve hiperwalk at call time, not import time.

    hiperwalk is provided by the [quvine] extra; require_module turns its absence
    into a message naming the extra and the install command rather than a bare
    ModuleNotFoundError. Resolving it at *import* time would be wrong: walks/base.py
    imports this module eagerly, so an RWR-only run -- which never touches a quantum
    walk -- would fail with a CTQW message pointing at the wrong dependency.
    """
    return require_module("hiperwalk", feature="continuous-time quantum walks (CTQW)")


def generate_ctqw_hiperwalk_scores(G, root, view_nodes=None, steps: int=20, time: float | None=None, gamma: float | None=None):
    """
    Return node probabilities from Hiperwalk continuous-time quantum walk (CTQW)
    Args:
        G (_type_): _description_
        root (_type_): _description_
        view_nodes (_type_, optional): _description_. Defaults to None.
        steps (int, optional): _description_. Defaults to 20.
        time (float | None, optional): _description_. Defaults to None.
        gamma (float | None, optional): _description_. Defaults to None.
    """
    if view_nodes is not None: 
        G = G.subgraph(view_nodes)
    if root not in G: 
        raise ValueError("Root node not in graph or view")
    
    nodes = list(G.nodes())
    node2i = {n:i for i,n in enumerate(nodes)}
    i2node = {i:n for n,i in node2i.items()}
    G_int = nx.relabel_nodes(G,node2i, copy=True)
    
    
    #build hiperwalk graph + ctqw
    hpw = _hiperwalk()
    hg = hpw.Graph(G_int)
    qw = hpw.ContinuousTime(graph=hg)
    
    if gamma is not None:
        qw.set_gamma(gamma)
    if time is not None:
        qw.set_time(time)
    
    root_i = node2i[root]
    state0 = qw.ket(root_i)
    
    #simulate to obtain final state at steps
    final_state = qw.simulate(range=(steps, steps+1), state=state0)
    
    #convert to node probability
    probs = qw.probability_distribution(final_state)
    probs = np.asarray(probs)[0]
    
    #map to original ids
    scores = {i2node[i]: float(probs[i]) for i in range(len(probs))}
    
    return scores


def generate_CTQW_walks(G, root, view_nodes=None, num_walks: int = 10, 
                                walk_length: int = 6, steps: int=20, 
                                time: float | None=None, 
                                gamma: float | None=None, 
                                rng=None):
    assert isinstance(rng, np.random.Generator)
    scores = generate_ctqw_hiperwalk_scores(G, 
                                            root=root, 
                                            view_nodes=view_nodes, 
                                            steps=steps, 
                                            time=time, 
                                            gamma=gamma)
    
    walks = sample_walks_from_distribution(scores, 
                                        num_walks=num_walks,
                                        walk_length=walk_length,
                                        rng=rng)
    return walks 
