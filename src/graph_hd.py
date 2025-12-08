import numpy as np
import networkx as nx
from tqdm import tqdm
from .hdc_ops import generate_item_memory, bind, bundle, cosine_similarity

class GraphHDClassifier:
    def __init__(self, dim=10000, max_nodes=100, centrality="pagerank"):
        self.dim = dim
        self.max_nodes = max_nodes
        self.centrality = centrality.lower()
        self.class_hypervectors = {} 
        self.positional_hvs = generate_item_memory(self.max_nodes, self.dim)
        
    def _compute_centrality(self, G):
        """Helper to compute centrality scores based on selected method."""
        try:
            if self.centrality == "pagerank":
                return nx.pagerank(G, alpha=0.85)
            elif self.centrality == "eigenvector":
                # Max iter increased for convergence stability
                return nx.eigenvector_centrality(G, max_iter=500, tol=1e-04)
            elif self.centrality == "degree":
                return nx.degree_centrality(G)
            elif self.centrality == "betweenness":
                return nx.betweenness_centrality(G)
            elif self.centrality == "closeness":
                return nx.closeness_centrality(G)
            elif self.centrality == "katz":
                 # Requires adjustment usually, straightforward call
                 return nx.katz_centrality(G, max_iter=1000, tol=1e-04)
            else:
                raise ValueError(f"Unknown centrality: {self.centrality}")
        except Exception:
            # Fallback for convergence failures (e.g. disconnected graph for eigenvector)
            return {n: 0 for n in G.nodes()}

    def encode_graph(self, G_obj):
        """
        Encodes a graph using Topological GraphHD.
        """
        nx_G = nx.Graph()
        nx_G.add_edges_from(G_obj.edges)
        
        # Ensure all nodes are added even if isolated (if we knew node count)
        # But G_obj usually only has edges.
        
        if nx_G.number_of_nodes() == 0:
             return np.zeros(self.dim)
             
        # Compute Centrality
        scores = self._compute_centrality(nx_G)
            
        # Sort nodes by Score (descending)
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Map Node -> Positional Hypervector
        node_vectors = {}
        for rank, (node, score) in enumerate(sorted_nodes):
            idx = rank % self.max_nodes
            node_vectors[node] = self.positional_hvs[idx]
        
        # Encode Edges
        edge_hvs = []
        for u, v in G_obj.edges:
            hv_u = node_vectors.get(u)
            hv_v = node_vectors.get(v)
            
            if hv_u is None or hv_v is None:
                continue
                
            val = bind(hv_u, hv_v)
            edge_hvs.append(val)
            
        if not edge_hvs:
             return np.zeros(self.dim)
             
        graph_hv = bundle(np.array(edge_hvs))
        
        return graph_hv
    
    def fit(self, graphs, labels):
        print(f"Encoding graphs for training ({self.centrality})...")
        class_accumulators = {}
        for G, label in tqdm(zip(graphs, labels), total=len(graphs), desc="Training"):
            ghv = self.encode_graph(G)
            if label not in class_accumulators:
                class_accumulators[label] = []
            class_accumulators[label].append(ghv)
            
        self.class_hypervectors = {}
        for label, hvs in class_accumulators.items():
            self.class_hypervectors[label] = bundle(np.array(hvs))

    def predict(self, graphs):
        predictions = []
        for G in tqdm(graphs, desc="Inference"):
            ghv = self.encode_graph(G)
            best_sim = -float('inf')
            best_label = None
            for label, centroid in self.class_hypervectors.items():
                sim = cosine_similarity(ghv, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            predictions.append(best_label)
        return predictions
