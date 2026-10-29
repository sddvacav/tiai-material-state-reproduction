"""CUDA device check shared by the GPU analyses."""
from __future__ import annotations
import torch

GPU_RECEIPT = {}

def gpu_device(device='cuda:0'):
    requested = torch.device(device)
    if requested.type == 'cpu':
        return requested
    if requested.type != 'cuda':
        raise ValueError('Statistics device must be explicitly CPU or CUDA')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: these statistics require CUDA')
    if not GPU_RECEIPT:
        device = requested
        probe = torch.arange(4096, dtype=torch.float64, device=device).square().sum()
        torch.cuda.synchronize(device)
        GPU_RECEIPT.update(device=torch.cuda.get_device_name(device), backend='torch CUDA', tensor_device=str(probe.device), probe_result=float(probe.item()))
    return requested
