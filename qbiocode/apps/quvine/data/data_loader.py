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

import pandas as pd 
import numpy as np
import networkx as nx 
import json

def load_graph(cfg):
    """
    Loads the largest connected component (giant component) of a network from a CSV file.

    """
    
    df = pd.read_csv(cfg.graph.path, dtype={'node1': str, 'node2': str})[["node1", "node2"]]
    graph = nx.from_pandas_edgelist(df, source="node1", target="node2")
    
    # Find the largest connected component in the graph
    giant = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(giant)
    
    return graph

def load_pegasus_results(path, disease):
    data = pd.read_csv(
        f"{path}/gwas_gene_pvals/{disease}/filtered_ncbi_PEGASUS_{disease}_gwas_data.csv")
    data = data[~data["NCBI_id"].isna()]
    data["NCBI_id"] = data["NCBI_id"].astype(str)
    
    pegasus_scores = {}
    for i, row in data.iterrows():
        pv = row["Pvalue"] if row["Pvalue"]>0.0 else row["Error"]
        pegasus_scores[row["NCBI_id"]] = np.maximum(1e-16, -np.log10(pv))
    data["Score"] = data["NCBI_id"].map(pegasus_scores)
    return data

def load_seeds_and_targets(path, disease):
    # with open("processed_data/gene_seeds/{}_seeds_gene2ncbi.json".format(disease), "r") as f:
    #     disease_seeds_gene2ncbi = json.load(f)
    # gene_seeds_ncbi = [str(n) for n in disease_seeds_gene2ncbi.values()]
    with open(path+"/gene_seeds/{}_ncbi_seeds.json".format(disease), "r") as f:
        disease_seeds_ncbi = json.load(f)
    gene_seeds_ncbi = [str(n) for n in disease_seeds_ncbi]
    # disease_seeds_gene2ncbi

    # Load targets
    with open(path+"/gwas_catalog_targets/{}_targets_gene2ncbi.json".format(disease), "r") as f:
        catalog_targets_gene2ncbi = json.load(f)
    ncbi_targets = list(catalog_targets_gene2ncbi.values())
    return gene_seeds_ncbi, ncbi_targets

def load_gwas_data(cfg, graph): 
    pegasus_data = load_pegasus_results(cfg.data_path, cfg.disease.name)
    seeds, targets = load_seeds_and_targets(cfg.data_path, cfg.disease.name)
    targets = [x for x in graph.nodes if x in targets]
    pegasus_scores = dict(zip(pegasus_data['NCBI_id'], pegasus_data['Score']))
    pagerank_seeds = {node: pegasus_scores.get(node,0) for node in graph.nodes}
    #seed2gene = dict(zip(pegasus_data['NCBI_id'], pegasus_data['Gene']))
    source = [x for x,y in pagerank_seeds.items() if y>0.3*pd.Series(pagerank_seeds).max()]
    source_genes = [x for x in source if x not in seeds]
    if cfg.verbose: 
        print(f"Number of seed genes: {len(source_genes)}, and target genes: {len(targets)}")
        
    return source_genes, targets
    