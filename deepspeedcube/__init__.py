import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_size(x: torch.Tensor) -> int:
    return x.element_size() * x.numel()
