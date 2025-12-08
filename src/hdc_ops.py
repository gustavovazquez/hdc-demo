import numpy as np

def generate_item_memory(n_items, dim):
    """
    Generates a memory of random bipolar hypervectors.
    
    Args:
        n_items (int): Number of items to generate.
        dim (int): Dimension of the hypervectors.
        
    Returns:
        np.ndarray: Matrix of shape (n_items, dim) with values {-1, 1}.
    """
    return np.random.choice([-1, 1], size=(n_items, dim))

def bind(hv1, hv2):
    """
    Performs the binding operation (XOR in bipolar domain, equivalent to multiplication).
    
    Args:
        hv1 (np.ndarray): First hypervector(s).
        hv2 (np.ndarray): Second hypervector(s).
        
    Returns:
        np.ndarray: Result of binding.
    """
    return hv1 * hv2

def bundle(hv_list):
    """
    Performs the bundling operation (superposition).
    Sum and then threshold (majority vote) to return to bipolar.
    
    Args:
        hv_list (np.ndarray): List or matrix of hypervectors to bundle.
        
    Returns:
        np.ndarray: Bundled hypervector.
    """
    # Sum along the 0-th axis (assuming hv_list is a matrix where each row is a vector)
    sum_hv = np.sum(hv_list, axis=0)
    
    # Thresholding to return to {-1, 1}
    # If sum > 0 -> 1, if sum < 0 -> -1. Randomly break ties (0).
    bundled = np.ones_like(sum_hv)
    bundled[sum_hv < 0] = -1
    
    # Handle ties (zeros) randomly
    zero_indices = (sum_hv == 0)
    if np.any(zero_indices):
        bundled[zero_indices] = np.random.choice([-1, 1], size=np.sum(zero_indices))
        
    return bundled

def cosine_similarity(hv1, hv2):
    """
    Calculates Cosine Similarity between two hypervectors.
    For bipolar vectors, this is proportional to the normalized element-wise dot product.
    
    Args:
        hv1 (np.ndarray): First hypervector.
        hv2 (np.ndarray): Second hypervector.
        
    Returns:
        float: Cosine similarity (-1 to 1).
    """
    # Norm of a bipolar vector of dim D is sqrt(D)
    # CosSim = (A . B) / (|A| * |B|) = (A . B) / D
    # for single vectors
    
    if hv1.ndim == 1 and hv2.ndim == 1:
        dot_product = np.dot(hv1, hv2)
        dim = hv1.shape[0]
        return dot_product / dim
    else:
        # Handle batch cases if needed, but for now assuming simple vectors or broadcasting
        # Standard cosine similarity
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(hv1, hv2) / (norm1 * norm2)
