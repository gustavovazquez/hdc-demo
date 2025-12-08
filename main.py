import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.data import load_tudataset
from src.graph_hd import GraphHDClassifier
import time

def run_dataset(name):
    print(f"\nProcessing {name}...")
    print("-" * (len(name) + 12))
    
    # 1. Load Data
    try:
        graphs, labels = load_tudataset(name)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return
        
    if not graphs:
        print(f"No graphs found for {name}.")
        return

    print(f"Loaded {len(graphs)} graphs.")
    
    # Calculate max_nodes for this dataset to size HDC memory
    max_nodes = 0
    for G in graphs:
        # TUDataset nodes are 1-based, usually contiguous per graph
        # But we need number of nodes in the graph
        # Our Graph object stores edges. 
        # Number of nodes ~ max(u, v) in edges?
        # Or we can count unique nodes in edges.
        nodes = set()
        for u, v in G.edges:
            nodes.add(u)
            nodes.add(v)
        if nodes:
            max_nodes = max(max_nodes, len(nodes))
            
    # Add a safety buffer or minimum
    max_nodes = max(max_nodes, 100) 
    print(f"Max nodes in dataset: {max_nodes}")
    
    # 2. Split Data
    # Stratify might fail if some class has only 1 member.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        # Fallback without stratify
        X_train, X_test, y_train, y_test = train_test_split(
            graphs, labels, test_size=0.2, random_state=42
        )
    
    # 3. Initialize Model with dynamic max_nodes
    model = GraphHDClassifier(dim=10000, max_nodes=max_nodes)
    
    # 4. Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 5. Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # 6. Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy [{name}]: {acc * 100:.2f}%")
    print(f"Train Time: {train_time:.2f}s | Inf Time: {inference_time:.2f}s")
    
    return {
        "dataset": name,
        "accuracy": acc,
        "train_time": train_time,
        "inference_time": inference_time
    }

def main():
    datasets = ["MUTAG", "DD", "ENZYMES", "NCI1", "PROTEINS", "PTC_FM"]
    results = []
    
    for ds in datasets:
        res = run_dataset(ds)
        if res:
            results.append(res)
            
    print("\nFinal Benchmark Results")
    print("=======================")
    print(f"{'Dataset':<15} | {'Accuracy':<10} | {'Train(s)':<10} | {'Inf(s)':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['dataset']:<15} | {r['accuracy']*100:6.2f}%    | {r['train_time']:8.2f}   | {r['inference_time']:8.2f}")

if __name__ == "__main__":
    main()

