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

import numpy as np  
import random
from qbiocode.apps.quvine.views.views import constrained_view

class ViewBuilder:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        
    def build(self, G, root): 
    
        views = constrained_view(G=G, 
                                root=root, 
                                num_views=self.cfg.views.num_views, 
                                max_nodes=self.cfg.views.max_nodes, 
                                max_edges=self.cfg.views.max_edges,
                                max_degree=self.cfg.views.max_degree,
                                rng=self.rng, 
                                degree_norm=self.cfg.views.degree_norm,
                                degree_alpha=self.cfg.views.degree_alpha
                                )    
    
        return views
