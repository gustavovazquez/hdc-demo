
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data import load_tudataset
from src.graph_hd import GraphHDClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

def run_experiment(dataset_name, metrics):
    print(f"\n>>> Processing Dataset: {dataset_name} <<<")
    
    try:
        graphs, labels = load_tudataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    if not graphs:
        return None

    # Calculate Max Nodes (for memory sizing)
    max_nodes = 0
    for G in graphs:
        nodes = set()
        for u, v in G.edges:
            nodes.add(u)
            nodes.add(v)
        if nodes:
            max_nodes = max(max_nodes, len(nodes))
    max_nodes = max(max_nodes, 100)
    print(f"Max nodes: {max_nodes}")

    # Determine Splitting Strategy
    # Using stratify if possible (better for comparison)
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
            # Skip heavy metrics for large graphs/datasets to save time if needed
            # DD has 5000 nodes. Betweenness is O(NM). It might hang.
            # We'll set a timeout or just try.
            # For this run, we will try all.
            
            model = GraphHDClassifier(dim=10000, max_nodes=max_nodes, centrality=metric)
            
            start = time.time()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            end = time.time()
            
            acc = accuracy_score(y_test, y_pred)
            results[metric] = acc
            print(f"Acc: {acc*100:.2f}% ({end-start:.2f}s)")
            
        except Exception as e:
            print(f"Failed: {e}")
            results[metric] = 0.0
            
    return results

def plot_results(all_results):
    datasets = list(all_results.keys())
    metrics = list(next(iter(all_results.values())).keys())
    
    # Prepare data for plotting
    # Structure: metric -> [score_ds1, score_ds2, ...]
    metric_scores = {m: [] for m in metrics}
    
    for ds in datasets:
        for m in metrics:
            metric_scores[m].append(all_results[ds].get(m, 0.0) * 100)
            
    # Plotting
    x = np.arange(len(datasets))
    width = 0.15
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for attribute, measurement in metric_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1
        
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('GraphHD Performance by Centrality Measure')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right', ncols=len(metrics))
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('benchmark_plot.png')
    print("\nPlot saved to benchmark_plot.png")

def main():
    datasets = ["MUTAG", "PTC_FM", "ENZYMES", "PROTEINS", "NCI1", "DD"]
    # Removed closeness/betweenness for DD usually? 
    # Let's keep them but user beware.
    metrics = ["pagerank", "eigenvector", "degree", "betweenness", "closeness"]
    
    all_results = {}
    
    for ds in datasets:
        # For DD/NCI1, maybe skip heavy metrics if slow?
        # NCI1 is small nodes (max 100), fine.
        # DD is large nodes (max 5000). Betweenness O(V*E) will be hours.
        # We will dynamically remove slow metrics for DD.
        current_metrics = metrics.copy()
        if ds == "DD":
            print(f"Skipping Betweenness/Closeness for {ds} due to size constraints.")
            current_metrics = ["pagerank", "eigenvector", "degree"]
            
        res = run_experiment(ds, current_metrics)
        if res:
            all_results[ds] = res
            
    # Print Table
    print("\nFinal Benchmark Results (Accuracy %)")
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
        
    # Plot
    plot_results(all_results)

if __name__ == "__main__":
    main()
