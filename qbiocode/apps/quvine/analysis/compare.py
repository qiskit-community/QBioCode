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

import itertools
from qbiocode.apps.quvine.analysis.analyze import procrustes_residual, cca_correlation, rsa_corr, knn_overlap, spectral_info


def compare_embeddings(store, cca_components=10, knn_k=10):
    """
    store: EmbeddingStore
    Returns dict: pair_name -> metrics
    """
    names = store.names()
    combos = itertools.combinations(names, 2)

    results = {}

    for a, b in combos:
        Z1 = store.get(a)
        Z2 = store.get(b)

        results[f"{a}_vs_{b}"] = {
            "procrustes": procrustes_residual(Z1, Z2),
            "cca_corr": cca_correlation(Z1, Z2, n_components=cca_components),
            "rsa_corr": rsa_corr(Z1, Z2),
            "knn_overlap": knn_overlap(Z1, Z2, k=knn_k),
        }
        
        all_embeddings = [] 
        for name in names: 
            all_embeddings.append(store.get(name))
            
        results['spectral'] = spectral_info(all_embeddings, names)

    return results
