import torch
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Mac M1/M2 users just in case
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

def generate_item_memory(n_items, dim, device=DEVICE):
    """
    Generates random bipolar hypervectors on the GPU.
    Values: {-1, 1}
    """
    # Generate 0 or 1, then map: 0->-1, 1->1
    mem = torch.randint(0, 2, (n_items, dim), device=device, dtype=torch.float32)
    mem[mem == 0] = -1
    return mem

def bind(hv1, hv2):
    """
    Element-wise multiplication (XOR in bipolar space).
    Supports broadcasting.
    """
    return hv1 * hv2

def bundle(hv_tensor):
    """
    Superposition (Sum) + Thresholding (Majority Vote).
    hv_tensor: (Batch, Dim) -> Bundled: (Dim,)
    """
    # Sum along column axis
    sum_hv = torch.sum(hv_tensor, dim=0)
    
    # Threshold
    bundled = torch.ones_like(sum_hv)
    bundled[sum_hv < 0] = -1
    
    # Handle ties randomly
    zeros = (sum_hv == 0)
    if zeros.any():
        # Random -1 or 1
        random_vals = torch.randint(0, 2, zeros.sum().size(), device=hv_tensor.device).float()
        random_vals[random_vals == 0] = -1
        bundled[zeros] = random_vals
        
    return bundled

def cosine_similarity(hv1, hv2):
    """
    Computes cosine similarity (normalized dot product).
    """
    if hv1.dim() == 1:
        hv1 = hv1.unsqueeze(0) # (1, D)
    if hv2.dim() == 1:
        hv2 = hv2.unsqueeze(0) # (1, D)
        
    # Normalize (L2 norm)
    # For bipolar vectors of dim D, norm is sqrt(D).
    # But usually we just use built-in cosine_similarity or manual dot/norm
    
    hv1_norm = torch.nn.functional.normalize(hv1, p=2, dim=1)
    hv2_norm = torch.nn.functional.normalize(hv2, p=2, dim=1)
    
    # Matrix multiplication: (Batch1, D) @ (D, Batch2) -> (Batch1, Batch2)
    # If standard 1-to-1:
    return torch.mm(hv1_norm, hv2_norm.t())
