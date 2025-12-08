import os
import requests
import zipfile
import io
import numpy as np
from collections import defaultdict

class Graph:
    def __init__(self, edges=None, node_labels=None, graph_label=None):
        self.edges = edges if edges is not None else []
        self.node_labels = node_labels if node_labels is not None else {}
        self.graph_label = graph_label

    def number_of_nodes(self):
        nodes = set()
        for u, v in self.edges:
            nodes.add(u)
            nodes.add(v)
        # Also include isolated nodes if they have labels
        nodes.update(self.node_labels.keys())
        return len(nodes)

def load_tudataset(name="MUTAG", root="data"):
    """
    Downloads and loads a TUDataset.
    
    Args:
        name (str): Name of the dataset (e.g., MUTAG, ENZYMES).
        root (str): Root directory to store the dataset.
        
    Returns:
        tuple: (graphs, labels)
            graphs: List of Graph objects.
            labels: List of graph labels.
    """
    dataset_dir = os.path.join(root, name)
    if not os.path.exists(dataset_dir):
        print(f"Downloading {name} dataset...")
        url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(root)
        print("Download and extraction complete.")
        
    # File paths
    # Note: TUDataset files usually have prefix {name}_
    # But sometimes inside the folder it's just the files.
    # MUTAG.zip extracts to a folder 'MUTAG/' containing 'MUTAG_A.txt' etc.
    
    prefix = os.path.join(dataset_dir, name)
    
    # helper to read file
    def read_file(suffix, dtype=int):
        path = f"{prefix}_{suffix}.txt"
        if os.path.exists(path):
            return np.loadtxt(path, dtype=dtype, delimiter=',') 
        return None

    # Load Adjacency Matrix (sparse)
    # Format: node_id_1, node_id_2
    # Indices are 1-based usually
    edges_raw = np.loadtxt(f"{prefix}_A.txt", dtype=int, delimiter=',')
    
    # Load Graph Indicator
    # Line i corresponds to node_id i (1-based), value is graph_id (1-based)
    graph_indicator = np.loadtxt(f"{prefix}_graph_indicator.txt", dtype=int)
    
    # Load Graph Labels
    graph_labels_raw = np.loadtxt(f"{prefix}_graph_labels.txt", dtype=int)
    
    # Load Node Labels (Optional)
    node_labels_raw = None
    if os.path.exists(f"{prefix}_node_labels.txt"):
        node_labels_raw = np.loadtxt(f"{prefix}_node_labels.txt", dtype=int)
        
    # Process data into Graph objects
    
    # Map graph_id -> list of nodes
    # graph_indicator has length = total_num_nodes
    # content: [1, 1, 1, 2, 2, ...] meaning first 3 nodes belong to graph 1
    
    # Warning: TUDataset indices are 1-based.
    # We will normalize node IDs to be local to each graph (0-based) for easier handling,
    # or keep them global. GraphHD 'encode_graph' handles edges (u,v).
    # If we keep global IDs, we might have a huge memory for 'node_memory'.
    # HDC usually encodes 'structure', so local IDs (canonical) or node features.
    # MUTAG has node labels (atom types). Is node identity important?
    # No, usually isomorphism invariant.
    # GraphHD uses node labels if available. If not, it relies on structural ID?
    # Actually, if we use node labels, the ID itself (u, v) matters for structure binding.
    # But (u, v) in GraphHD paper usually means binding unique ID of u with unique ID of v?
    # Wait, if all Carbon atoms have same vector, then C-C bond is always same vector.
    # That loses structural info (which C is connected to which C).
    # GraphHD usually adds a unique ID *and* label, or uses PageRank.
    # For now, let's keep it simple: Use the global node ID (unique) combined with label.
    # But GraphHD paper says: "We assign a hypervector to each vertex... initialized with random...".
    # If we use global ID, each node in the DATASET has a unique vector. That works.
    
    # Create Graph objects
    unique_graph_ids = np.unique(graph_indicator)
    graphs_dict = {gid: Graph() for gid in unique_graph_ids}
    
    # 1. Assign Graph Indices
    # graph_indicator index i (0-based in array) corresponds to node_id i+1 (1-based in file)
    # Let's map global node ID (1-based) -> graph_id
    node_to_graph = {}
    for i, gid in enumerate(graph_indicator):
        node_id = i + 1
        node_to_graph[node_id] = gid
        # Add node specific label if exists
        if node_labels_raw is not None:
             graphs_dict[gid].node_labels[node_id] = node_labels_raw[i]
        
    # 2. Add Edges
    for u, v in edges_raw:
        # u, v are global 1-based indices
        gid = node_to_graph.get(u)
        # Assuming edges are only within the same graph
        if gid:
            graphs_dict[gid].edges.append((u, v))
            
    # 3. Assign Graph Labels
    # graph_labels_raw corresponds to unique_graph_ids in order?
    # usually yes, line i is label for graph i (if graphs are 1..N)
    # verify
    graphs = []
    labels = []
    
    # TUDataset graph IDs usually start at 1 and are contiguous
    sorted_gids = sorted(graphs_dict.keys())
    
    for i, gid in enumerate(sorted_gids):
        g = graphs_dict[gid]
        g.graph_label = graph_labels_raw[i]
        graphs.append(g)
        labels.append(g.graph_label)
        
    return graphs, labels

if __name__ == "__main__":
    gs, ls = load_tudataset()
    print(f"Loaded {len(gs)} graphs.")
    print(f"First graph edges: {len(gs[0].edges)}")
