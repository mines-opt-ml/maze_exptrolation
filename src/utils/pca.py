import numpy as np
import torch

from src.utils.device import get_device

def get_pca(X, n=2):
    """ Project rows of data matrix X onto n principal components using SVD """

    # Flatten data
    X = X.reshape(X.shape[0], -1)

    # Convert to torch on device
    X = torch.tensor(X, dtype=torch.float32, device=get_device())

    with torch.no_grad():
        # Center the data
        X = X - torch.mean(X, axis=0)

        # Compute the SVD
        U, S, V = torch.linalg.svd(X, full_matrices=False)

        # Project the data onto the first n principal components
        Y = torch.mm(X, V[:n].T)

    # Convert back to numpy
    Y = Y.cpu().numpy()

    return Y

