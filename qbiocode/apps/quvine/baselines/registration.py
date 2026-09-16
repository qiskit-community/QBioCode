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
Method Registration for QuVINE Registry

This module provides the registration function that registers all 32 baseline methods
with the method registry using the existing config builders and adapters.
"""

import logging
from .registry import MethodRegistry, MethodMetadata
from .configs import (
    build_node2vec_config,
    build_appnp_config,
    build_graphsage_config,
    build_netmf_config,
    build_gcnmf_config,
    build_baseline_filter_config,
    build_quvine_filter_config,
    build_gat_config,
    build_graphgps_config,
    build_quvine_sgns_config,
)
from .adapters import (
    run_node2vec_adapter,
    run_appnp_adapter,
    run_graphsage_adapter,
    run_netmf_adapter,
    run_baseline_filter_adapter,
    run_quvine_heat_adapter,
    run_quvine_poly_adapter,
    run_baseline_gcnmf_adapter,
    run_quvine_sgns_adapter,
    run_gat_adapter,
    run_graphgps_adapter,
)

logger = logging.getLogger(__name__)


def register_all_methods(registry: MethodRegistry) -> None:
    """
    Register all 32 baseline methods with the registry.
    
    This function registers:

    - 5 Baselines: node2vec, appnp, graphsage, netmf, baseline_gcnmf
    - 10 Filter variants: baseline_filter_heat/poly, filter_rwr_heat/poly,
      filter_ctqw_heat/poly, filter_dtqw_heat/poly
    - 10 GAT variants: gat_baseline, gat_heat, gat_poly, gat_rwr (classical),
      gat_ctqw_heat/poly, gat_dtqw_heat/poly, gat_rwr_heat/poly (qcal)
    - 10 GraphGPS variants: graphgps_baseline, graphgps_heat, graphgps_poly,
      graphgps_rwr, graphgps_ctqw_heat/poly, graphgps_dtqw_heat/poly,
      graphgps_rwr_heat/poly
    - 3 SGNS: quvine_ctqw, quvine_dtqw, quvine_rwr
    
    Args:
        registry: MethodRegistry instance to register methods with
    """
    
    # ========== 5 BASELINES ==========
    
    registry.register(MethodMetadata(
        name="node2vec",
        config_builder=build_node2vec_config,
        executor=run_node2vec_adapter,
        category="baseline",
        description="Node2Vec random walk embeddings"
    ))
    
    registry.register(MethodMetadata(
        name="appnp",
        config_builder=build_appnp_config,
        executor=run_appnp_adapter,
        category="baseline",
        description="APPNP (Approximate Personalized Propagation of Neural Predictions)"
    ))
    
    registry.register(MethodMetadata(
        name="graphsage",
        config_builder=build_graphsage_config,
        executor=run_graphsage_adapter,
        category="baseline",
        description="GraphSAGE neighborhood sampling"
    ))
    
    registry.register(MethodMetadata(
        name="netmf",
        config_builder=build_netmf_config,
        executor=run_netmf_adapter,
        category="baseline",
        description="NetMF matrix factorization"
    ))
    
    registry.register(MethodMetadata(
        name="baseline_gcnmf",
        config_builder=lambda cfg, seed: build_gcnmf_config(cfg, seed, "baseline_gcnmf"),
        executor=run_baseline_gcnmf_adapter,
        category="baseline",
        description="GCN-MF baseline"
    ))
    
    # ========== 10 FILTER VARIANTS ==========
    
    # Baseline filters (heat and poly)
    registry.register(MethodMetadata(
        name="baseline_filter_heat",
        config_builder=lambda cfg, seed: build_baseline_filter_config(cfg, seed, filter_type="heat"),
        executor=run_baseline_filter_adapter,
        category="baseline",
        description="Baseline heat kernel filter"
    ))
    
    registry.register(MethodMetadata(
        name="baseline_filter_poly",
        config_builder=lambda cfg, seed: build_baseline_filter_config(cfg, seed, filter_type="poly"),
        executor=run_baseline_filter_adapter,
        category="baseline",
        description="Baseline polynomial filter"
    ))
    
    # RWR filters (heat and poly)
    registry.register(MethodMetadata(
        name="filter_rwr_heat",
        config_builder=lambda cfg, seed: build_quvine_filter_config(cfg, seed, "filter_rwr", "heat"),
        executor=run_quvine_heat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="RWR with heat kernel filter"
    ))
    
    registry.register(MethodMetadata(
        name="filter_rwr_poly",
        config_builder=lambda cfg, seed: build_quvine_filter_config(cfg, seed, "filter_rwr", "poly"),
        executor=run_quvine_poly_adapter,
        requires_q_targets=True,
        category="quantum",
        description="RWR with polynomial filter"
    ))
    
    # CTQW filters (heat and poly)
    registry.register(MethodMetadata(
        name="filter_ctqw_heat",
        config_builder=lambda cfg, seed: build_quvine_filter_config(cfg, seed, "filter_ctqw", "heat"),
        executor=run_quvine_heat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="CTQW with heat kernel filter"
    ))
    
    registry.register(MethodMetadata(
        name="filter_ctqw_poly",
        config_builder=lambda cfg, seed: build_quvine_filter_config(cfg, seed, "filter_ctqw", "poly"),
        executor=run_quvine_poly_adapter,
        requires_q_targets=True,
        category="quantum",
        description="CTQW with polynomial filter"
    ))
    
    # DTQW filters (heat and poly)
    registry.register(MethodMetadata(
        name="filter_dtqw_heat",
        config_builder=lambda cfg, seed: build_quvine_filter_config(cfg, seed, "filter_dtqw", "heat"),
        executor=run_quvine_heat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="DTQW with heat kernel filter"
    ))
    
    registry.register(MethodMetadata(
        name="filter_dtqw_poly",
        config_builder=lambda cfg, seed: build_quvine_filter_config(cfg, seed, "filter_dtqw", "poly"),
        executor=run_quvine_poly_adapter,
        requires_q_targets=True,
        category="quantum",
        description="DTQW with polynomial filter"
    ))
    
    # ========== 10 GAT VARIANTS ==========

    registry.register(MethodMetadata(
        name="gat_baseline",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_baseline"),
        executor=run_gat_adapter,
        category="baseline",
        description="Baseline GAT (Graph Attention Network)"
    ))

    # GAT classical single-axis controls (no quantum calibration)
    registry.register(MethodMetadata(
        name="gat_heat",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_heat"),
        executor=run_gat_adapter,
        category="classical",
        description="GAT with fixed heat-kernel filtered features (classical)"
    ))

    registry.register(MethodMetadata(
        name="gat_poly",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_poly"),
        executor=run_gat_adapter,
        category="classical",
        description="GAT with fixed polynomial filtered features (classical)"
    ))

    registry.register(MethodMetadata(
        name="gat_rwr",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_rwr"),
        executor=run_gat_adapter,
        category="classical",
        description="GAT with RWR-diffused features (classical)"
    ))

    # GAT + CTQW (heat and poly)
    registry.register(MethodMetadata(
        name="gat_ctqw_heat",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_ctqw_heat"),
        executor=run_gat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GAT with CTQW heat kernel"
    ))
    
    registry.register(MethodMetadata(
        name="gat_ctqw_poly",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_ctqw_poly"),
        executor=run_gat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GAT with CTQW polynomial"
    ))
    
    # GAT + DTQW (heat and poly)
    registry.register(MethodMetadata(
        name="gat_dtqw_heat",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_dtqw_heat"),
        executor=run_gat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GAT with DTQW heat kernel"
    ))
    
    registry.register(MethodMetadata(
        name="gat_dtqw_poly",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_dtqw_poly"),
        executor=run_gat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GAT with DTQW polynomial"
    ))
    
    # GAT + RWR (heat and poly)
    registry.register(MethodMetadata(
        name="gat_rwr_heat",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_rwr_heat"),
        executor=run_gat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GAT with RWR heat kernel"
    ))
    
    registry.register(MethodMetadata(
        name="gat_rwr_poly",
        config_builder=lambda cfg, seed: build_gat_config(cfg, seed, "gat_rwr_poly"),
        executor=run_gat_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GAT with RWR polynomial"
    ))
    
    # ========== 10 GRAPHGPS VARIANTS ==========

    registry.register(MethodMetadata(
        name="graphgps_baseline",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_baseline"),
        executor=run_graphgps_adapter,
        category="baseline",
        description="Baseline GraphGPS"
    ))

    # GraphGPS classical single-axis controls (no quantum calibration)
    registry.register(MethodMetadata(
        name="graphgps_heat",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_heat"),
        executor=run_graphgps_adapter,
        category="classical",
        description="GraphGPS with fixed heat-kernel filtered features (classical)"
    ))

    registry.register(MethodMetadata(
        name="graphgps_poly",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_poly"),
        executor=run_graphgps_adapter,
        category="classical",
        description="GraphGPS with fixed polynomial filtered features (classical)"
    ))

    registry.register(MethodMetadata(
        name="graphgps_rwr",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_rwr"),
        executor=run_graphgps_adapter,
        category="classical",
        description="GraphGPS with RWR-diffused features (classical)"
    ))

    # GraphGPS + CTQW (heat and poly)
    registry.register(MethodMetadata(
        name="graphgps_ctqw_heat",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_ctqw_heat"),
        executor=run_graphgps_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GraphGPS with CTQW heat kernel"
    ))
    
    registry.register(MethodMetadata(
        name="graphgps_ctqw_poly",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_ctqw_poly"),
        executor=run_graphgps_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GraphGPS with CTQW polynomial"
    ))
    
    # GraphGPS + DTQW (heat and poly)
    registry.register(MethodMetadata(
        name="graphgps_dtqw_heat",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_dtqw_heat"),
        executor=run_graphgps_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GraphGPS with DTQW heat kernel"
    ))
    
    registry.register(MethodMetadata(
        name="graphgps_dtqw_poly",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_dtqw_poly"),
        executor=run_graphgps_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GraphGPS with DTQW polynomial"
    ))
    
    # GraphGPS + RWR (heat and poly)
    registry.register(MethodMetadata(
        name="graphgps_rwr_heat",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_rwr_heat"),
        executor=run_graphgps_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GraphGPS with RWR heat kernel"
    ))
    
    registry.register(MethodMetadata(
        name="graphgps_rwr_poly",
        config_builder=lambda cfg, seed: build_graphgps_config(cfg, seed, "graphgps_rwr_poly"),
        executor=run_graphgps_adapter,
        requires_q_targets=True,
        category="quantum",
        description="GraphGPS with RWR polynomial"
    ))
    
    # ========== 3 SGNS (QuVINE walks) ==========
    
    registry.register(MethodMetadata(
        name="quvine_ctqw",
        config_builder=lambda cfg, seed: build_quvine_sgns_config(cfg, seed, "quvine_ctqw", "ctqw"),
        executor=run_quvine_sgns_adapter,
        requires_q_targets=False,
        category="quantum",
        description="QuVINE SGNS embedding with CTQW walks"
    ))

    registry.register(MethodMetadata(
        name="quvine_dtqw",
        config_builder=lambda cfg, seed: build_quvine_sgns_config(cfg, seed, "quvine_dtqw", "dtqw"),
        executor=run_quvine_sgns_adapter,
        requires_q_targets=False,
        category="quantum",
        description="QuVINE SGNS embedding with DTQW walks"
    ))

    registry.register(MethodMetadata(
        name="quvine_rwr",
        config_builder=lambda cfg, seed: build_quvine_sgns_config(cfg, seed, "quvine_rwr", "rwr"),
        executor=run_quvine_sgns_adapter,
        requires_q_targets=False,
        category="quantum",
        description="QuVINE SGNS embedding with RWR walks"
    ))
    
    logger.info(f"Registered all 36 methods: {len(registry)} total")
    baseline_count = sum(1 for m in registry.list_methods() if (meta := registry.get_metadata(m)) and meta.category == 'baseline')
    classical_count = sum(1 for m in registry.list_methods() if (meta := registry.get_metadata(m)) and meta.category == 'classical')
    quantum_count = sum(1 for m in registry.list_methods() if (meta := registry.get_metadata(m)) and meta.category == 'quantum')
    logger.info(f"  - {baseline_count} baseline methods")
    logger.info(f"  - {classical_count} classical methods")
    logger.info(f"  - {quantum_count} quantum methods")

