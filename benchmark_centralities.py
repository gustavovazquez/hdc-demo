
import time
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data import load_tudataset
from src.graph_hd import GraphHDClassifier
import networkx as nx

# Suppress sklearn/numpy warnings for clean output
warnings.filterwarnings("ignore")

def run_centrality_test(graphs, labels, metric, max_nodes):
    print(f"\n--- Testing Metric: {metric.upper()} ---")
    
    # Stratified Split (same for all metrics to be fair)
    # We set random_state to ensure same split for each metric run
    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    model = GraphHDClassifier(dim=10000, max_nodes=max_nodes, centrality=metric)
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    inf_time = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    
    return {
        "metric": metric,
        "accuracy": acc,
        "train_time": train_time,
        "inf_time": inf_time
    }

def main():
    dataset_name = "MUTAG"
    print(f"Benchmarking Centrality Measures on {dataset_name}")
    print("==========================================")
    
    graphs, labels = load_tudataset(dataset_name)
    
    # Calculate Max Nodes
    max_nodes = 0
    for G in graphs:
        nodes = set()
        for u, v in G.edges:
            nodes.add(u)
            nodes.add(v)
        if nodes:
            max_nodes = max(max_nodes, len(nodes))
    max_nodes = max(max_nodes, 100)
    print(f"Max Nodes: {max_nodes}\n")
    
    metrics = ["pagerank", "eigenvector", "betweenness", "closeness", "degree"]
    
    results = []
    
    for m in metrics:
        try:
            res = run_centrality_test(graphs, labels, m, max_nodes)
            results.append(res)
        except Exception as e:
            print(f"Failed {m}: {e}")
            
    print("\nSummary of Results (MUTAG)")
    print("==========================")
    print(f"{'Metric':<15} | {'Accuracy':<10} | {'Train(s)':<10}")
    print("-" * 40)
    
    # Sort by Accuracy Descending
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    
    for r in results:
        print(f"{r['metric']:<15} | {r['accuracy']*100:6.2f}%    | {r['train_time']:8.4f}")

if __name__ == "__main__":
    main()
