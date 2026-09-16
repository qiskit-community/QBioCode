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

from qbiocode.apps.quvine.walks.rwr import generate_RWR_pagerank_walks
from qbiocode.apps.quvine.walks.ctqw import generate_CTQW_walks
from qbiocode.apps.quvine.walks.dtqw import generate_DTQW_walks
import numpy as np 

WALKS={
    "rwr": generate_RWR_pagerank_walks,
    "ctqw": generate_CTQW_walks,
    "dtqw": generate_DTQW_walks,
}

class BaseWalker:
    """
    Base class for all walk types.

    Subclasses must implement `_generate_walks`.
    """

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.kinds = list(cfg.walks.kinds)
        
        assert isinstance(rng, np.random.Generator)
        
        unknown = set(self.kinds) - WALKS.keys() 
        if unknown: 
            raise ValueError(f"Unknown walk kinds: {unknown}")

        
    def run(self, graph, root, view_nodes=None):
        """Run the walk.

        Args:
            graph (networkx graph): subgraph of the views
            root (string): Root node tied to the views
            view_nodes (list of list): list of nodes in the views

        Returns:
            Dict[str, List[walks]]
        """
        
        out = {} 
        for kind in self.kinds: 
            out[kind] = self._run_walk(kind, graph, root, view_nodes)
        
        return out 
    
    def _run_walk(self, kind, graph, root, view_nodes=None): 
        
        if kind == "rwr": 
            return generate_RWR_pagerank_walks(
            G=graph, 
            root=root, 
            view_nodes=view_nodes,
            num_walks=self.cfg.walks.num_walks,
            walk_length=self.cfg.walks.walk_length, 
            restart_prob=self.cfg.walks.restart_prob,
            max_iter=self.cfg.walks.max_iter, 
            rng=self.rng
            )
        elif kind == "ctqw": 
            return generate_CTQW_walks(
            G=graph, 
            root=root, 
            view_nodes=view_nodes,
            num_walks=self.cfg.walks.num_walks,
            walk_length=self.cfg.walks.walk_length, 
            time=self.cfg.walks.time,
            steps=self.cfg.walks.steps,
            rng=self.rng
            )
        elif kind == "dtqw": 
            return generate_DTQW_walks(
            G=graph, 
            root=root, 
            view_nodes=view_nodes,
            num_walks=self.cfg.walks.num_walks,
            walk_length=self.cfg.walks.walk_length, 
            steps=self.cfg.walks.steps,
            coin=self.cfg.walks.coin, 
            rng=self.rng
            )
