
import time
import numpy as np
import networkx as nx
import torch
import warnings
from src.data import load_tudataset
from src.graph_hd import GraphHDClassifier
from src.graph_hd_torch import GraphHDTorch
from src.hdc_ops_torch import generate_item_memory, bind, bundle, cosine_similarity, DEVICE

warnings.filterwarnings("ignore")

def profile_component_breakdown(graphs, centrality_method="pagerank"):
    print(f"\n--- Component Latency Analysis (Single Core, 50 graphs) ---")
    
    # Measure Centrality Time vs Linear Scan vs HDC Ops
    t_centrality = 0
    t_sort = 0
    t_hdc = 0
    
    # Pre-gen memory
    dim = 10000
    max_nodes = 100
    mem = np.random.randn(max_nodes, dim).astype(np.float32) # Mock
    
    sample_graphs = graphs[:50]
    
    start_total = time.time()
    
    for G_obj in sample_graphs:
        # Reconstruct (overhead)
        nx_G = nx.Graph()
        nx_G.add_edges_from(G_obj.edges)
        if nx_G.number_of_nodes() == 0: continue
        
        # 1. Centrality
        t0 = time.time()
        if centrality_method == "pagerank":
            scores = nx.pagerank(nx_G, alpha=0.85)
        elif centrality_method == "degree":
            scores = nx.degree_centrality(nx_G)
        t1 = time.time()
        t_centrality += (t1 - t0)
        
        # 2. Sort/Map
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        node_vectors = {}
        for rank, (node, score) in enumerate(sorted_nodes):
            idx = rank % max_nodes
            node_vectors[node] = mem[idx]
        t2 = time.time()
        t_sort += (t2 - t1)
        
        # 3. HDC Bind/Bundle (Numpy)
        edge_hvs = []
        for u, v in G_obj.edges:
            hv_u = node_vectors.get(u)
            hv_v = node_vectors.get(v)
            if hv_u is not None and hv_v is not None:
                edge_hvs.append(hv_u * hv_v)
        
        if edge_hvs:
            s = np.sum(edge_hvs, axis=0)
            _ = np.where(s < 0, -1, 1)
            
        t3 = time.time()
        t_hdc += (t3 - t2)
        
    total_measured = t_centrality + t_sort + t_hdc
    print(f"Breakdown for {len(sample_graphs)} graphs (Centrality: {centrality_method}):")
    print(f"  Centrality: {t_centrality:.4f}s ({t_centrality/total_measured*100:.1f}%)")
    print(f"  Sort/Map:   {t_sort:.4f}s ({t_sort/total_measured*100:.1f}%)")
    print(f"  HDC Ops:    {t_hdc:.4f}s ({t_hdc/total_measured*100:.1f}%)")


def profile_model(model_name, model_class, graphs, labels, n_jobs=None):
    print(f"\n--- Profiling {model_name} ---")
    
    kwargs = {'dim': 10000, 'max_nodes': 100, 'centrality': "pagerank"}
    if n_jobs is not None:
        kwargs['n_jobs'] = n_jobs
        
    model = model_class(**kwargs)
        
    # Profile Encoding (for fit)
    print("Profiling Encoding/Fit...")
    start_fit = time.time()
    try:
        model.fit(graphs, labels)
    except Exception as e:
        print(f"Fit failed: {e}")
        return
    end_fit = time.time()
    print(f"Fit Time: {end_fit - start_fit:.4f}s")
    
    return end_fit - start_fit

def main():
    dataset = "PROTEINS" 
    print(f"Loading {dataset}...")
    try:
        graphs, labels = load_tudataset(dataset)
    except Exception as e:
        print(f"Could not load {dataset}: {e}")
        return

    print(f"Loaded {len(graphs)} graphs.")
    
    # 1. Breakdown Analysis
    profile_component_breakdown(graphs, centrality_method="pagerank")
    
    # 2. Baseline Profile
    print("\nRunning Baseline (Sequential)...")
    t_base = profile_model("GraphHD (Baseline CPU)", GraphHDClassifier, graphs, labels)
    
    # 3. Parallel Profile
    print("\nRunning Parallel (Joblib/GPU)...")
    t_torch = profile_model("GraphHDTorch (MP/GPU)", GraphHDTorch, graphs, labels, n_jobs=-1)
    
    if t_base and t_torch:
        print(f"\nSpeedup: {t_base / t_torch:.2f}x")


if __name__ == "__main__":
    main()
