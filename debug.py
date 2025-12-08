from src.data import load_tudataset
from src.graph_hd import GraphHDClassifier
from src.hdc_ops import cosine_similarity, generate_item_memory, bind, bundle
import numpy as np

def debug():
    # 1. Test Ops
    print("Testing Ops...")
    v1 = generate_item_memory(1, 10000).flatten()
    v2 = generate_item_memory(1, 10000).flatten()
    print(f"v1[:10]: {v1[:10]}")
    print(f"Bind(v1, v2)[:10]: {bind(v1, v2)[:10]}")
    print(f"Sim(v1, v2): {cosine_similarity(v1, v2)}")
    
    # 2. Test Data
    print("\nTesting Data...")
    graphs, labels = load_tudataset("MUTAG")
    g1 = graphs[0]
    print(f"G1 Nodes: {len(g1.node_labels)}")
    print(f"G1 Edges: {len(g1.edges)}")
    print(f"G1 Sample Node Labels: {list(g1.node_labels.items())[:5]}")
    print(f"G1 Edge Sample: {g1.edges[:5]}")
    
    # 3. Test Encoding
    print("\nTesting Encoding...")
    model = GraphHDClassifier(dim=10000)
    
    hv1 = model.encode_graph(g1)
    print(f"HV1 Mean: {np.mean(hv1)}")
    print(f"HV1[:20]: {hv1[:20]}")
    
    g2 = graphs[1]
    hv2 = model.encode_graph(g2)
    print(f"HV2 Mean: {np.mean(hv2)}")
    print(f"Sim(HV1, HV2): {cosine_similarity(hv1, hv2)}")
    
    if np.array_equal(hv1, hv2):
        print("HV1 and HV2 are IDENTICAL!")
    else:
        print("HV1 and HV2 are DIFFERENT.")

if __name__ == "__main__":
    debug()
