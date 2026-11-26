"""The device a model predicts on: CUDA, else Apple MPS, else CPU. Shared by src.rnn and src.transformer."""
import torch


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
