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

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import time
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from qbiocode.apps.quvine.data.data_loader import load_graph, load_gwas_data
from qbiocode.apps.quvine.data.prepare import PrepareGraphConfig, prepare_graph
from qbiocode.apps.quvine.api.sgns import run_sgns
from qbiocode.apps.quvine.api.targets import build_quantum_targets
from qbiocode.apps.quvine.embedding.registry import EmbeddingStore
from qbiocode.apps.quvine.analysis.compare import compare_embeddings
from qbiocode.apps.quvine.analysis.analyze import normalize
# get_stats, draw_graph, plot_metric, plot_precision_recall, plot_metric_vs_k
# live in quvine.utils.utilities (imported via the wildcard import below).
# Registry system imports
from qbiocode.apps.quvine.baselines.registry import MethodRegistry
from qbiocode.apps.quvine.baselines.registration import register_all_methods
from qbiocode.apps.quvine.baselines.hyperparameter_loader import HyperparameterLoader, set_global_loader
from qbiocode.apps.quvine.fusion.fuse import fuse_embeddings
from qbiocode.apps.quvine.evaluation.ranking import (
    seed_centroid_scores,
    max_seed_cosine_scores,
    evaluate_embeddings_ranking
    )   
from qbiocode.apps.quvine.utils.seed import set_global_seed
from qbiocode.apps.quvine.utils.utilities import *


class Pipeline: 
    """
    End-to-end quvine pipeline. 
    
    Stages: 
    Graph Loading 
    Seed/Target Loading 
    Preprocessing
    View Building
    Walking
    Embedding Training 
    Evaluation
    
    """
    
    def __init__(self, cfg:DictConfig): 
        self.cfg = cfg
        self.log = logging.getLogger(self.__class__.__name__)
        self.run_dir = Path(cfg.runtime.output_dir)
        if self.run_dir.exists():
            if self.cfg.verbose: 
                print(f"Directory {self.run_dir} exists")
        else: 
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.n_iters = cfg.experiment.iterations 
        self.base_seed = cfg.experiment.base_seed
        
    def run(self): 
        self.log.info("Pipeline started (%d iterations)", self.n_iters)
        
        #load graph data once
        graph_data = self._load_graph()
        if self.cfg.verbose: 
            print(get_stats(graph_data))
        
        if self.cfg.gwas_target:
            source, target = self._load_gwas_data(graph_data)
        else:
            source = None
            target = None
        
        ## Preprocess graph 
        graph_data = self._preprocess_graph( 
                                            graph_data, 
                                            source, 
                                            target)
        
        if self.cfg.draw.graph: 
            draw_graph(cfg=self.cfg, 
                    G=graph_data, 
                    source=source, 
                    target=target)
        
        all_results = []

        for it in range(self.n_iters):
            self.log.info("Iteration %d / %d", it + 1, self.n_iters)
            self._set_iteration_seed(it)

            res = self._run_single_iteration(it, graph_data, source, target)
            all_results.append(res)
        
        if self.cfg.evaluation.enabled:
            # process and save evaluation results
            self._save_evaluation_results(all_results=all_results, nodes=list(graph_data.nodes))  
        
        if self.cfg.save_embeddings:
            # save and output embeddings
            self._save_embeddings(all_results=all_results)   
            
        

    #-----------------
    # One iteration
    # ----------------
    
    def _run_single_iteration(self, it, graph_data, source, target):

        beg_time = time.time()

        # SGNS walk embeddings (views -> walks -> corpus -> word2vec).
        # Shared with the quvine.embed() API via the extracted SGNS core.
        embeddings = run_sgns(
            self.cfg,
            graph_data,
            it,
            kinds=list(self.cfg.walks.kinds),
            n_jobs=self.cfg.runtime.n_jobs,
            chunk_size=self.cfg.runtime.chunk_size,
        )

        store = EmbeddingStore()
        for name, Z in embeddings.items():
            store.add(name, Z)
        end_time = time.time()
        time_taken = end_time - beg_time
        if self.cfg.verbose:
            print(f"Time taken for one QuVINE iteration {time_taken/60} minutes")
        
        ## baselines and quantum-calibrated downstream methods
        q_targets = None
        if source is not None and len(source) > 0:
            max_support = getattr(self.cfg.baselines, "quantum_target_max_nodes", 64)
            q_targets = build_quantum_targets(graph_data, source, max_support=max_support)

        # Initialize hyperparameter loader if dataset name is available
        dataset_name = getattr(self.cfg.data, 'name', None)
        if dataset_name:
            # Determine tuning directory based on dataset type
            if hasattr(self.cfg.data, 'ppi') and self.cfg.data.ppi:
                tuning_dir = "ppi_tuning_by_task"
            elif hasattr(self.cfg.data, 'realworld') and self.cfg.data.realworld:
                tuning_dir = "realworld_tuning_by_task"
            else:
                tuning_dir = "tuning_by_task"
            
            loader = HyperparameterLoader(
                tuning_dir=tuning_dir,
                dataset_name=dataset_name
            )
            set_global_loader(loader)
            if self.cfg.verbose:
                print(f"Loaded hyperparameter tuning from {tuning_dir}/{dataset_name}")
        
        # Create and populate method registry
        beg_time = time.time()
        registry = MethodRegistry(self.cfg, base_seed=self.base_seed, verbose=self.cfg.verbose)
        register_all_methods(registry)
        
        if self.cfg.verbose:
            print(f"Registered {len(registry)} methods")
            print(f"  - {len(registry.list_methods('baseline'))} baseline methods")
            print(f"  - {len(registry.list_methods('quantum'))} quantum methods")
        
        # Run all enabled methods
        results = registry.run_all(
            graph_data=graph_data,
            q_targets=q_targets,
            store=store
        )
        
        # Print summary
        end_time = time.time()
        total_time = end_time - beg_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        if self.cfg.verbose:
            print(f"\n{'='*60}")
            print(f"Baseline Methods Summary:")
            print(f"  Total methods run: {len(results)}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Total time: {total_time/60:.2f} minutes")
            print(f"{'='*60}\n")
            
            # Print individual method times
            for result in results:
                if result.success:
                    print(f"  ✓ {result.name}: {result.execution_time/60:.2f} minutes")
                else:
                    print(f"  ✗ {result.name}: FAILED - {result.error}")
        ## compare embeddings
        if self.cfg.analysis.enabled:
            comparison_metrics = compare_embeddings(
                                        store,
                                        cca_components=self.cfg.analysis.cca_components,
                                        knn_k=self.cfg.analysis.knn_k,
                                        )
        else: 
            comparison_metrics = None
        
        ## fuse embeddings
        if self.cfg.fusion.enabled:
            
            beg_time = time.time()
            
            L = nx.normalized_laplacian_matrix(G=graph_data,
                                        nodelist=graph_data.nodes).toarray().astype(np.float32)
            
            fused_list, fuse_metric = fuse_embeddings(
                                        store,
                                        method=self.cfg.fusion.method,
                                        k=self.cfg.fusion.k,
                                        L=L
                                    )

            for i, Z_fused in enumerate(fused_list): 
                store.add(fuse_metric[i], Z_fused)

            end_time = time.time() 
            time_taken = end_time - beg_time
            if self.cfg.verbose:
                print(f"Time taken for fusion {time_taken/60} minutes")
        
        
        ## target prioritization evaluation 
        if self.cfg.evaluation.enabled: 
            
            seed_indices = [
                i for i, node in enumerate(graph_data.nodes)
                if node in source
            ]
            scores_by_method = {}
            for name, Z in store.items():
                if self.cfg.evaluation.centroid:
                    scores_by_method[f"{name}_centroid"] = seed_centroid_scores(
                        Z, seed_indices
                    )
                if self.cfg.evaluation.max_seed:
                    scores_by_method[f"{name}_max"] = max_seed_cosine_scores(
                        Z, seed_indices
                    )

            ranking_df = evaluate_embeddings_ranking(
                scores_by_method=scores_by_method,
                subgraph=graph_data,
                seeds=source,
                targets=target,
                nodes=graph_data.nodes,
                k_values=self.cfg.evaluation.k_values,
                n_repeats=self.cfg.evaluation.n_repeats,
                deg_tol=self.cfg.evaluation.deg_tol,
                iteration=it,
            )
            # standard metadata for analysis 
            
            return {
                    "iteration": it,
                    "ranking_df": ranking_df,
                    "comparison": comparison_metrics
                }
        else: 
            return {
                "iteration": it, 
                "embeddings": store, 
                "nodes": list(graph_data.nodes), 
                "comparison": comparison_metrics
            }

    #-----------------
    # Preprocess
    # ----------------
    
    def _preprocess_graph(self, graph_data, source, target):
        cfg_pg = PrepareGraphConfig(
                            subsample_nodes=self.cfg.preprocess.subsample.enabled, 
                            max_nodes=self.cfg.preprocess.subsample.max_nodes, 
                            radius=self.cfg.preprocess.subsample.radius,
                            sparsify_edges=self.cfg.preprocess.sparsify.enabled,
                            retain_ratio=self.cfg.preprocess.sparsify.retain_ratio,
                            max_degree=self.cfg.preprocess.sparsify.max_degree,
                            scoring=self.cfg.preprocess.sparsify.scoring,
                            verbose=self.cfg.verbose
                            )
        graph_data = prepare_graph(
                            cfg_pg, 
                            graph=graph_data, 
                            seeds=source, 
                            targets=target, 
                            seed=self.cfg.seed
                            )
        return graph_data 
    
    #-----------------
    # Data Loading
    # ----------------
    
    def _load_graph(self):
        self.log.info("Loading graph: %s", self.cfg.graph.name)
        
        return load_graph(self.cfg)
    
    def _load_gwas_data(self, graph_data):
        self.log.info("Loading gwas data: %s", self.cfg.disease.name)
        return load_gwas_data(self.cfg, graph_data)
    
    def _set_iteration_seed(self, it):
        seed = self.base_seed + it
        set_global_seed(seed)
        self.log.debug("Iteration seed set to %d", seed)

    # The SGNS walk/view/corpus/word2vec core and quantum-target construction
    # now live in quvine.api (sgns.py, targets.py) and are shared with embed().

    def _save_evaluation_results(self, all_results, nodes):
        
        ranking_df = self._post_process_ranking(all_results)
        comparison_df = self._post_process_comparison(all_results)
            
        out_dir = HydraConfig.get().runtime.output_dir
        os.makedirs(out_dir, exist_ok=True)

        self.log.info("Saving outputs to %s", out_dir)
        
        ranking_path = os.path.join(out_dir, "ranking_results.csv")
        ranking_df.to_csv(ranking_path, index=False)

        comparison_path = os.path.join(out_dir, "embedding_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        
        cfg_path = os.path.join(out_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
            
        summary = {
                "n_iterations": self.n_iters,
                "n_nodes": len(nodes),
                "walks": OmegaConf.to_container(self.cfg.walks.kinds, resolve=True),
                }

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if self.cfg.plots:
            self._plot_all(
                    ranking_df=ranking_df, 
                    out_dir=out_dir
                    )
        self.log.info("All results saved to %s", out_dir)
        
    def _post_process_ranking(self, all_results):
        
        ranking_dfs = [
                        r["ranking_df"] for r in all_results
                        if r["ranking_df"] is not None
                    ]   

        ranking_results_df = pd.concat(
            ranking_dfs,
            ignore_index=True
        )
        
        return ranking_results_df 
    
    def _post_process_comparison(self, all_results): 
        comparison_rows = []

        for r in all_results:
            it = r["iteration"]
            for pair, metrics in r["comparison"].items():
                for name, value in metrics.items():
                    comparison_rows.append({
                        "iteration": it,
                        "pair": pair,
                        "metric": name,
                        "value": value,
                    })

        comparison_df = pd.DataFrame(comparison_rows)

        return comparison_df
                
    def _plot_all(self, ranking_df, out_dir):
        
        plot_metric(cfg=self.cfg, 
                        df=ranking_df, 
                        metric='recall', 
                        file_path=out_dir)
        plot_metric(cfg=self.cfg, 
                    df=ranking_df, 
                    metric='precision', 
                    file_path=out_dir)
        
        plot_precision_recall(df=ranking_df, 
                            control='true', 
                            file_path=out_dir)
        plot_precision_recall(df=ranking_df, 
                            control='degree_matched', 
                            file_path=out_dir)
        plot_precision_recall(df=ranking_df, 
                            control='distance_matched', 
                            file_path=out_dir)
        
        plot_metric_vs_k(df=ranking_df, 
                        metric='recall',
                        control='true',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='precision',
                        control='true',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='recall',
                        control='degree_matched',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='precision',
                        control='degree_matched',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='recall',
                        control='distance_matched',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='precision',
                        control='distance_matched',
                        file_path=out_dir)

    def _save_embeddings(self, all_results):

        out_dir = HydraConfig.get().runtime.output_dir
        emb_dir = os.path.join(out_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)

        self.log.info("Saving embeddings to %s", emb_dir)

        comparison_df = self._post_process_comparison(all_results=all_results)
        comparison_df.to_csv(os.path.join(emb_dir, "embedding_comparison.csv"), index=False)

        for res in all_results:
            iter_num = res["iteration"]

            npz_payload = {
                emb_name: emb.astype(np.float32, copy=False)
                for emb_name, emb in res["embeddings"].items()
                if emb is not None
            }
            npz_payload["nodes"] = np.asarray(res["nodes"])

            np.savez_compressed(
                os.path.join(emb_dir, f"embeddings_iter_{iter_num}.npz"),
                **npz_payload,
            )

            self.log.debug(
                "Saved iteration %d embeddings: %s",
                iter_num,
                list(npz_payload.keys()),
            )
