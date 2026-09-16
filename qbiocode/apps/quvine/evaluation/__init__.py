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

from qbiocode.apps.quvine.evaluation.ranking import evaluate_embeddings_ranking
from qbiocode.apps.quvine.evaluation.classification import (
    generate_community_labels,
    generate_degree_labels,
    generate_centrality_labels,
    generate_core_periphery_labels,
    evaluate_node_classification,
    evaluate_all_label_strategies,
    summarize_classification_results
)
from qbiocode.apps.quvine.evaluation.link_prediction import (
    sample_negative_edges,
    split_edges,
    compute_edge_features,
    evaluate_link_prediction,
    evaluate_link_prediction_cv,
    evaluate_all_edge_feature_methods,
    summarize_link_prediction_results,
    compute_structural_link_features
)

__all__ = [
    # Ranking
    "evaluate_embeddings_ranking",
    # Classification
    "generate_community_labels",
    "generate_degree_labels",
    "generate_centrality_labels",
    "generate_core_periphery_labels",
    "evaluate_node_classification",
    "evaluate_all_label_strategies",
    "summarize_classification_results",
    # Link Prediction (Enhanced with hard negatives and inner product/cosine)
    "sample_negative_edges",  # Now supports 'random', 'hard_2hop', 'same_community'
    "split_edges",
    "compute_edge_features",  # Now supports 'inner_product' and 'cosine'
    "evaluate_link_prediction",
    "evaluate_link_prediction_cv",
    "evaluate_all_edge_feature_methods",
    "summarize_link_prediction_results",
    "compute_structural_link_features",
]