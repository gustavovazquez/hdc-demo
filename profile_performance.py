
import time
import numpy as np
import torch
import warnings
from src.data import load_tudataset
from src.graph_hd import GraphHDClassifier
from src.graph_hd_torch import GraphHDTorch

warnings.filterwarnings("ignore")

def profile_model(model_name, model_class, graphs, labels, n_jobs=None):
    print(f"\n--- Profiling {model_name} ---")
    
    # Init
    if n_jobs is not None:
        model = model_class(dim=10000, max_nodes=100, centrality="degree", n_jobs=n_jobs)
    else:
        model = model_class(dim=10000, max_nodes=100, centrality="degree")
        
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
    
    # Profile Inference
    print("Profiling Prediction...")
    # Duplicate graphs to increase load if needed, but let's stick to dataset
    start_pred = time.time()
    try:
        model.predict(graphs) # Predict on training set just for perf
    except Exception as e:
        print(f"Predict failed: {e}")
        return
    end_pred = time.time()
    print(f"Predict Time: {end_pred - start_pred:.4f}s")
    
    total_time = (end_fit - start_fit) + (end_pred - start_pred)
    print(f"Total Cycle Time: {total_time:.4f}s")
    return total_time

def main():
    dataset = "PROTEINS" 
    print(f"Loading {dataset}...")
    try:
        graphs, labels = load_tudataset(dataset)
    except Exception as e:
        print(f"Could not load {dataset}: {e}")
        return

    print(f"Loaded {len(graphs)} graphs.")
    
    # Profile Baseline
    # time_base = profile_model("GraphHD (Baseline CPU)", GraphHDClassifier, graphs, labels)
    
    # Profile Torch/MP
    # n_jobs = -1 (all cores)
    time_torch = profile_model("GraphHDTorch (MP/GPU)", GraphHDTorch, graphs, labels, n_jobs=-1)
    
    # Explicit single core test for Torch to see overhead?
    # time_torch_1 = profile_model("GraphHDTorch (1 Core)", GraphHDTorch, graphs, labels, n_jobs=1)


if __name__ == "__main__":
    main()
