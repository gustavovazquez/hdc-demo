import numpy as np
import networkx as nx
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from .hdc_ops_torch import generate_item_memory, bind, bundle, cosine_similarity, DEVICE

def process_single_graph(G_edges, centrality_method, max_nodes, positional_hvs_cpu, dim):
    """
    Worker function to process a single graph on CPU.
    We pass edges instead of full object to minimize pickle overhead, 
    reconstructing nx.Graph inside.
    
    Returns: numpy array of the Graph Hypervector.
    """
    # Reconstruct Graph
    nx_G = nx.Graph()
    nx_G.add_edges_from(G_edges)
    
    if nx_G.number_of_nodes() == 0:
        return np.zeros(dim)

    # Compute Centrality
    try:
        if centrality_method == "pagerank":
            scores = nx.pagerank(nx_G, alpha=0.85)
        elif centrality_method == "eigenvector":
            scores = nx.eigenvector_centrality(nx_G, max_iter=1000, tol=1e-04)
        elif centrality_method == "degree":
            scores = nx.degree_centrality(nx_G)
        elif centrality_method == "betweenness":
            scores = nx.betweenness_centrality(nx_G)
        elif centrality_method == "closeness":
            scores = nx.closeness_centrality(nx_G)
        else:
            scores = {n: 0 for n in nx_G.nodes()}
    except:
        scores = {n: 0 for n in nx_G.nodes()}

    # Sort nodes
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Map Node ID -> Positional Vector (numpy)
    node_vectors = {}
    for rank, (node, score) in enumerate(sorted_nodes):
        idx = rank % max_nodes
        node_vectors[node] = positional_hvs_cpu[idx]

    # Encode Edges (HDC Bind)
    # We do binding in numpy here to save GPU transfer overhead for small individual ops,
    # or we could move to Torch. Given parallelism, Numpy on CPU is fine for individual items.
    # Then we bundle at the end.
    
    edge_hvs = []
    for u, v in G_edges:
        hv_u = node_vectors.get(u)
        hv_v = node_vectors.get(v)
        if hv_u is not None and hv_v is not None:
             # XOR (Bipolar multiplication)
             edge_hvs.append(hv_u * hv_v)
             
    if not edge_hvs:
        return np.zeros(dim, dtype=np.float32)
        
    # Bundle (Sum)
    # Aggregating for one graph
    sum_hv = np.sum(edge_hvs, axis=0)
    
    # Binarize
    graph_hv = np.ones_like(sum_hv, dtype=np.float32)
    graph_hv[sum_hv < 0] = -1
    # Ties handled simply for speed
    
    return graph_hv

class GraphHDTorch:
    def __init__(self, dim=10000, max_nodes=100, centrality="pagerank", n_jobs=-1):
        self.dim = dim
        self.max_nodes = max_nodes
        self.centrality = centrality.lower()
        self.n_jobs = n_jobs
        
        # We generate memory on GPU, but for the parallel CPU workers, we need a CPU copy
        self.positional_hvs_gpu = generate_item_memory(self.max_nodes, self.dim, device=DEVICE)
        self.positional_hvs_cpu = self.positional_hvs_gpu.cpu().numpy()
        
        self.class_hypervectors = {} # Will be on GPU

    def batch_encode(self, graphs):
        """
        Parallel encoding of a list of graphs.
        Returns tensor on GPU.
        """
        # Prepare list of edges (lighter to pickle)
        graphs_edges = [list(G.edges) for G in graphs]
        
        # Run Parallel Job
        # n_jobs=-1 uses all available cores
        results = Parallel(n_jobs=self.n_jobs, batch_size=32)(
            delayed(process_single_graph)(
                edges, 
                self.centrality, 
                self.max_nodes, 
                self.positional_hvs_cpu, 
                self.dim
            ) for edges in graphs_edges
        )
        
        # Convert list of numpy arrays to single Tensor on GPU
        return torch.tensor(np.array(results), dtype=torch.float32, device=DEVICE)

    def fit(self, graphs, labels):
        print(f"Encoding {len(graphs)} graphs (Parallel CPU)...")
        
        train_hvs = self.batch_encode(graphs) # (N_samples, Dim)
        
        # Aggregation by Class
        unique_labels = list(set(labels))
        self.class_hypervectors = {}
        
        print("Training Class Models (GPU)...")
        for lbl in unique_labels:
            # Find indices for this label
            indices = [i for i, x in enumerate(labels) if x == lbl]
            indices_tensor = torch.tensor(indices, device=DEVICE)
            
            # Select vectors
            class_vectors = torch.index_select(train_hvs, 0, indices_tensor)
            
            # Bundle
            self.class_hypervectors[lbl] = bundle(class_vectors)

    def predict(self, graphs):
        print(f"Inference on {len(graphs)} graphs...")
        
        # Encode Query Graphs
        query_hvs = self.batch_encode(graphs) # (N_query, Dim)
        
        # Stack Class Hypervectors into a matrix
        # Labels order
        labels_list = list(self.class_hypervectors.keys())
        class_matrix = torch.stack([self.class_hypervectors[l] for l in labels_list]) # (N_classes, Dim)
        
        # Compute Cosine Similarity Matrix
        # (N_query, Dim) @ (Dim, N_classes) -> (N_query, N_classes)
        # Using normalized cosine similarity helper
        
        # Normalize queries
        q_norm = torch.nn.functional.normalize(query_hvs, p=2, dim=1)
        # Normalize classes
        c_norm = torch.nn.functional.normalize(class_matrix, p=2, dim=1)
        
        # Sim Matrix
        sim_matrix = torch.mm(q_norm, c_norm.t()) # (N_query, N_classes)
        
        # Argmax
        best_indices = torch.argmax(sim_matrix, dim=1).cpu().numpy()
        
        predictions = [labels_list[idx] for idx in best_indices]
        return predictions
