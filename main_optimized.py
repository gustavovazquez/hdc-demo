import time
import os
import torch
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data import load_tudataset
from src.graph_hd_torch import GraphHDTorch
from src.hdc_ops_torch import DEVICE

warnings.filterwarnings("ignore")

def run_experiment(dataset_name, metrics):
    print(f"\n==================================================")
    print(f"Dataset: {dataset_name} | Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPUs available: {os.cpu_count()}")
    print(f"==================================================")
    
    try:
        graphs, labels = load_tudataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    if not graphs:
        return None

    # Calculate Max Nodes
    max_nodes = 0
    for G in graphs:
        if G.number_of_nodes() > 0:
            max_nodes = max(max_nodes, G.number_of_nodes())
    max_nodes = max(max_nodes, 100)
    print(f"Max nodes: {max_nodes} | Graphs: {len(graphs)}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=0.2, random_state=42
        )

    results = {}
    
    for metric in metrics:
        print(f"  > Metric: {metric}...", end=" ", flush=True)
        try:
            # Initialize Optimized Model
            model = GraphHDTorch(dim=10000, max_nodes=max_nodes, centrality=metric, n_jobs=-1)
            
            start = time.time()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            end = time.time()
            
            acc = accuracy_score(y_test, y_pred)
            results[metric] = acc
            print(f"Acc: {acc*100:.2f}% (Time: {end-start:.2f}s)")
            
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            results[metric] = 0.0
            
    return results

def main():
    datasets = ["MUTAG", "PTC_FM", "ENZYMES", "PROTEINS", "NCI1", "DD"]
    # Even for DD, with 16 cores, Betweenness might simpler be faster, 
    # but O(VE) is still brutal. We'll enable it to test the 16 cores power.
    # If it's too slow, user can kill it.
    metrics = ["pagerank", "eigenvector", "degree", "betweenness", "closeness"]
    
    all_results = {}
    
    for ds in datasets:
        current_metrics = metrics.copy()
        if ds == "DD":
            # Even with 16 cores, Betweenness on 5000 nodes is heavy.
            # Let's try skipping it initially to ensure user gets result.
            print(f"Skipping Betweenness/Closeness for {ds} (heuristic).")
            current_metrics = ["pagerank", "eigenvector", "degree"]
            
        res = run_experiment(ds, current_metrics)
        if res:
            all_results[ds] = res
            
    # Print Table
    print("\nFinal Optimized Results (Accuracy %)")
    print("====================================")
    header = f"{'Dataset':<10} | " + " | ".join([f"{m:<12}" for m in metrics])
    print(header)
    print("-" * len(header))
    
    for ds, res in all_results.items():
        row = f"{ds:<10} | "
        for m in metrics:
            val = res.get(m, "N/A")
            if isinstance(val, float):
                row += f"{val*100:6.2f}%      | "
            else:
                row += f"{val:<12} | "
        print(row)

if __name__ == "__main__":
    main()
