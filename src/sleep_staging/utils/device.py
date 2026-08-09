"""Backend selection."""
import torch


def get_device(requested=None):
    """Select requested backend, otherwise CUDA, MPS, then CPU."""
    if requested: return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
