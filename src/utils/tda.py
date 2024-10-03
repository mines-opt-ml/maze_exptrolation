""" 
Based on https://arxiv.org/abs/1704.08382
"""

import pickle
import numpy as np
import torch
import time
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.utils.device import get_device
from src.utils.loading import get_mazes, load_model

def get_diagram(X, dtype=np.float32, embed_dim=0, delay=1, max_homo=1, verbose=True):
    """ Get the persistence diagram for data X. """

    # Convert from torch tensor to numpy array if necessary
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    # Ensure full precision
    X = X.astype(dtype)

    if verbose:
        print(f'    Performing TDA with {embed_dim = }, {delay = }, and {X.shape = } ({X.nbytes/1e9:.3f}GB)')

    # Flatten dimensions beyond first
    X = X.reshape(X.shape[0], -1)
    if verbose:
        print(f'    Flattened {X.shape = }')

    # Reduce dimensionality of X using SVD, if first dimension is smaller
    F, P = X.shape
    if F < P:
        X = reduce(X)
        if verbose:
            print(f'    Reduced {X.shape = } ({X.nbytes/1e9:.3f}GB)')

    # Compute distance matrix of sliding window embedding of X
    distance_matrix = get_distance_matrix(X)
    max_distance = np.max(distance_matrix)
    if verbose:
        print(f'    Computed {distance_matrix.shape = } with {max_distance = :.3f}')

    # Use ripser to compute persistence diagram
    diagram = ripser(distance_matrix, maxdim=max_homo, coeff=2, distance_matrix=True)['dgms']

    return diagram, max_distance

def get_distance_matrix(X, embed_dim=0, delay=1, threshold=1e-10):
    """ 
    Get distance matrix of rows of sliding window embedding of flattened data X. 
    Distances smaller than threshold are set to zero.
    """

    # Ensure dimensions beyond first have been flattened
    assert len(X.shape) == 2

    # Window size must be less than total number of iterations
    assert (embed_dim+1)*delay < X.shape[0]

    if delay == 1:
        # Compute squared distance matrix for X
        squared_distance_matrix_X = np.square(squareform(pdist(X), checks=False))

        # Compute distance matrix for sliding window embedding of X
        d = embed_dim
        F = X.shape[0] # frames
        W = F - d # windows
        squared_distance_matrix_SW = np.zeros(shape=(W, W))
        for i in range(W):
            for j in range(i+1, W):
                for k in range(d+1):
                    squared_distance_matrix_SW[i, j] += squared_distance_matrix_X[i+k, j+k]
                squared_distance_matrix_SW[j, i] = squared_distance_matrix_SW[i, j]
        assert np.all(squared_distance_matrix_SW == squared_distance_matrix_SW.T)
        assert np.all(squared_distance_matrix_SW >= 0)
        distance_matrix_SW = np.sqrt(squared_distance_matrix_SW)

    else:
        raise NotImplementedError
    
    # Zero distances that are too small 
    distance_matrix_SW[distance_matrix_SW < threshold] = 0

    return distance_matrix_SW

def get_sw(X, embed_dim, delay, verbose=False):
    """
    Get normalized sliding window embedding of data, following
    (Quasi)Periodicity Quantification in Video Data by Tralie & Perea.
        embed_dim: d in their paper
        delay: tau in their paper
    """
    X = X.reshape(X.shape[0], -1)
    F, D = X.shape

    # Create sliding window embedding tensor
    window_size = embed_dim*delay
    SW = np.zeros((F - window_size, (embed_dim + 1) * D))
    for i in range(F - window_size):
        SW[i] = X[i : i + window_size + 1: delay].flatten()

    if verbose:
        print(f'{SW.shape = }')

    return SW

def reduce(X):
    """ Reduce dimensionality of X using SVD for memory efficiency """
    X = torch.from_numpy(X).to(get_device())
    with torch.no_grad():
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        X_reduced = U * S
    torch.cuda.empty_cache()
    return X_reduced.cpu().numpy()

def get_betti_nums(diagram, threshold):
    """ Calculate Betti numbers for persistence diagram D with given threshold """

    betti_nums = np.zeros(len(diagram), dtype=int)

    # If diagram corresponds to single point, return [1, 0]
    if (diagram[0] == [[0, np.inf]]).all() and (diagram[1] == np.zeros((0, 2))).all():
        betti_nums[0] = 1
        betti_nums[1] = 0

    # Otherwise, calculate persistent homologies above threshold
    else:
        max_death = get_max_death(diagram)
        for i in range(len(diagram)):
            for j in range(len(diagram[i])):
                lifetime = diagram[i][j, 1] - diagram[i][j, 0]
                if lifetime > threshold * max_death:
                    betti_nums[i] += 1

    return betti_nums

def get_max_death(diagram):
    """ Get maximum death in D, ignoring infinity """
    max_death = 0
    for i in range(len(diagram)):
        for j in range(len(diagram[i])):
            if diagram[i][j,1] != np.inf:
                max_death = max(max_death, diagram[i][j,1])
    return max_death

class Analysis:
    """
    Class for TDA analysis and results.
    """

    def __init__(self, maze_sizes, num_mazes, model_name, iters, dtype, embed_dim, delay, max_homo):
        self.maze_sizes = maze_sizes
        self.num_mazes = num_mazes
        self.model = load_model(model_name)
        self.iters = iters
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.delay = delay
        self.max_homo = max_homo
        self.times = np.zeros((len(maze_sizes), num_mazes), dtype=np.float32)
        self.diagrams = np.zeros((len(maze_sizes), num_mazes, 2), dtype=object)
        self.max_distances = np.zeros((len(maze_sizes), num_mazes), dtype=dtype)

    def get_name(self):
        name = f'{self.model.name()}'
        for attr, value in self.__dict__.items():
            if attr == 'iters':
                name += f'_{attr}-{min(value)},{max(value)}'
            elif attr == 'dtype':
                name += f'_dtype-{self.max_distances.dtype.name}'
            elif attr not in ['model', 'times', 'diagrams', 'max_distances']:
                name += f'_{attr}-{value}'
        return name
    
    def save(self):
        """ Save analysis object """
        file_name = f'outputs/tda/analysis/{self.get_name()}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def analyze(self, verbose=True):
        """ Perform TDA analysis on latent iterates of model while solving mazes """
        for i, maze_size in enumerate(self.maze_sizes):
            # Print summary
            if verbose:
                print(f'Analyzing {self.model.name()} on {self.num_mazes} mazes of size {maze_size} ...')

            # Load mazes
            start_time = time.time()
            inputs, _ = get_mazes(maze_size=maze_size, num_mazes=self.num_mazes)
            if verbose:
                print(f'    Loaded mazes in {time.time() - start_time:.2f}s')

            # Generate each latent series and perform TDA
            for j in range(self.num_mazes):
                # Generate latent series
                start_time = time.time()
                input = inputs[j:j+1]
                latent = self.model.input_to_latent(input)
                latent_series = self.model.latent_forward(latent, input, iters=self.iters)
                latent_series = latent_series.cpu().numpy()
                torch.cuda.empty_cache() 

                # Perform TDA
                diagram, max_distance = get_diagram(latent_series.squeeze(), self.dtype, self.embed_dim, self.delay, self.max_homo, verbose=False)
                self.diagrams[i, j, 0] = diagram[0]
                self.diagrams[i, j, 1] = diagram[1]
                self.max_distances[i, j] = max_distance

                # Save results
                self.save()
                
                self.times[i, j] = time.time() - start_time
                if verbose:
                    print(f'    Analyzed latent series for maze {j+1} of {self.num_mazes} in {self.times[i, j]:.2f}s')
            
        if verbose:
            print(f'Analysis complete after {np.sum(self.times):.2f}s')

    def get_betti_nums(self, threshold):
        """ Get Betti numbers for diagrams with given threshold """
        betti_nums = np.zeros((len(self.maze_sizes), self.num_mazes, self.max_homo+1), dtype=int)
        for i in range(len(self.maze_sizes)):
            for j in range(self.num_mazes):
                betti_nums[i, j] = get_betti_nums(self.diagrams[i, j], threshold)
        return betti_nums

    def print_time(self):
        """ Print time for analysis """
        print(f'Time for analysis: {np.sum(self.times)/60:.2f}min')
                