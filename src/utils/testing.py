import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from src.utils.seed import set_seed
from src.utils.loading import get_mazes, load_model

class Test:
    """
    Test object to evaluate the extrapolation accuracy of a model on mazes of different sizes or percolations.
    """
    def __init__(self, seed=0, model_name='dt_net', dataset='maze-dataset', num_mazes=10, iters=[30], maze_sizes=[9], percolations=[0.0], deadend_start=True, batch_size=10, verbose=True):
        self.seed = seed
        set_seed(seed)
        self.model = load_model(model_name)
        self.dataset = dataset
        self.num_mazes = num_mazes
        self.iters = iters
        self.maze_sizes = maze_sizes
        self.percolations = percolations
        self.deadend_start = deadend_start
        self.batch_size = batch_size
        self.verbose = verbose
        self.extrap_param_name, self.extrap_param = self.get_extrap_param()
        self.corrects = np.zeros((len(self.iters), len(self.extrap_param), self.num_mazes))
        self.accuracies = np.zeros((len(self.iters), len(self.extrap_param)))
        self.start_neighbors = np.zeros((len(self.extrap_param), num_mazes))
        self.times = np.zeros((len(self.iters), len(self.extrap_param)))

        # Check parameters are increasing
        if not all(iters[i] < iters[i + 1] for i in range(len(iters) - 1)):
            raise ValueError('Iterations should be increasing.')
        if not all(maze_sizes[i] < maze_sizes[i + 1] for i in range(len(maze_sizes) - 1)):
            raise ValueError('Maze sizes should be increasing.')
        if not all(percolations[i] < percolations[i + 1] for i in range(len(percolations) - 1)):
            raise ValueError('Percolations should be increasing.')

    def get_extrap_param(self):
        if len(self.maze_sizes) > 1 and len(self.percolations) == 1:
            self.percolation = self.percolations[0]
            if self.verbose:
                print(f'Evaluating extrapolation over maze sizes: {self.maze_sizes}')
            return 'maze_size', self.maze_sizes
        elif len(self.percolations) > 1 and len(self.maze_sizes) == 1:
            self.maze_size = self.maze_sizes[0]
            if self.verbose:
                print(f'Evaluating extrapolation over percolations: {self.percolations}')
            return 'percolation', self.percolations
        else:
            raise ValueError('Invalid parameters: Either maze_sizes or percolations should have multiple values, but not both.')
        
    def get_name(self):
        name = f'model-{self.model.name()}'
        for attr, value in self.__dict__.items():
            if attr in ['dataset', 'num_mazes', 'iters', 'maze_sizes', 'percolations', 'deadend_start']:
                name += f'_{attr}-{value}'
        return name
        
    def get_accuracies(self):
        """ 
        Compute the accuracy of the model on mazes of different sizes or percolations for different numbers of iterations.
        """

        # For each extrapolation parameter, get mazes, compute accuracy, and measure time
        for param in self.extrap_param:
            j = self.extrap_param.index(param)

            # Get mazes for current extrapolation parameter
            inputs, solutions = self.get_mazes(param)

            # Count start neighbors
            self.start_neighbors[j] = count_start_neighbors(inputs)

            # Batch test accuracy
            for b_start_idx in range(0, self.num_mazes, self.batch_size):
                b_end_idx = min(b_start_idx + self.batch_size, self.num_mazes)
                inputs_batch = inputs[b_start_idx:b_end_idx]
                solutions_batch = solutions[b_start_idx:b_end_idx]

                # Compute predictions and measure time 
                start_time = time.time()
                predictions_batch = self.model.predict(inputs_batch, self.iters)
                t = time.time() - start_time

                # Count correct predictions and time
                for iter in self.iters:
                    i = self.iters.index(iter)
                    self.corrects[i, j, b_start_idx:b_end_idx] = compare_mazes(predictions_batch[i], solutions_batch)               
                    self.times[i, j] += (iter / max(self.iters))*t

            # Compute accuracies
            self.accuracies[:, j] = self.corrects[:, j].mean(axis=1)

            if self.verbose:
                for iter in self.iters:
                    i = self.iters.index(iter)
                    t = self.times[i, j]
                    print(f"Model: {self.model.name()}, {self.extrap_param_name.capitalize()}: {param}, Iter: {iter}, Accuracy: {self.accuracies[i, j]}, Time: {t if t < 60 else t / 60:.2f}{'s' if t < 60 else 'min'}")

    def print(self):
        # Print test parameters
        print()
        print(f'Model: {self.model.name()}')
        for attr, value in self.__dict__.items():
            if attr in ['dataset', 'num_mazes', 'iters', 'maze_sizes', 'percolations', 'deadend_start']:
                print(f"{attr.capitalize()}: {value}")
        print(f'Time: {self.times[-1,:].sum() if self.times.sum() < 60 else self.times.sum() / 60:.2f}{"s" if self.times.sum() < 60 else "min"}')
        print()

        # Print accuracies
        for j, param in enumerate(self.extrap_param):
            for i, iter in enumerate(self.iters):
                accuracy = self.accuracies[i, j]
                t = self.times[i, j]
                print(f"{self.extrap_param_name.capitalize()}: {param}, Iter: {iter}, Accuracy: {accuracy}, Time: {t if t < 60 else t / 60:.2f}{'s' if t < 60 else 'min'}")

    def get_mazes(self, param):
        if self.extrap_param_name == 'maze_size':
            return get_mazes(self.dataset, maze_size=param, num_mazes=self.num_mazes, gen='dfs_perc', percolation=self.percolation, deadend_start=self.deadend_start)
        elif self.extrap_param_name == 'percolation':
            return get_mazes(self.dataset, maze_size=self.maze_size, num_mazes=self.num_mazes, gen='dfs_perc', percolation=param, deadend_start=self.deadend_start)

def compare_mazes(mazes_1, mazes_2):
    """
    Compare two batches of mazes of the same size,
    and return tensor of boolean values indicating which mazes match.
    """

    return [torch.all(mazes_1[i] == mazes_2[i]).item() for i in range(mazes_1.shape[0])]

def count_start_neighbors_nonparallel(inputs):
    """
    Count the number of neighbors of the starting position in each maze.
    """

    batch_size = inputs.shape[0]
    pixels = inputs.shape[-1]
    assert inputs.shape == (batch_size, 3, pixels, pixels)

    # For each batch_idx, determine index of start position (i.e. upper left green pixel)
    start_positions = -np.ones((batch_size, 2), dtype=int)
    for batch_idx in range(batch_size):
        for i in range(3, (pixels-3)-1, 2):
            for j in range(3, (pixels-3)-1, 2):
                if (inputs[batch_idx, :, i, j] == torch.tensor([0, 1, 0]).to(inputs.device)).all().item():
                    start_positions[batch_idx] = np.array([i, j], dtype=int)
                    break
            if start_positions[batch_idx,0] != -1:
                break
    # Each start index should be upper left pixel of a node
    assert np.all((start_positions-3) % 2 == 0)

    # Count neighbors of the starting position
    neighbors = np.zeros(batch_size)
    for batch_idx in range(batch_size):
        i, j = start_positions[batch_idx]
        for di, dj in [(-2,0), (2,0), (0,-2), (0,2)]:
            # A non-black adjacent pixel indicates a neighbor
            neighbors[batch_idx] += inputs[batch_idx, :, i+di, j+dj].any().item()

    return neighbors

def count_start_neighbors(inputs):
    """
    Count the number of neighbors of the starting position in each maze.
    """
    batch_size, _, pixels, _ = inputs.shape
    assert inputs.shape == (batch_size, 3, pixels, pixels)

    # Determine the start positions (upper left green pixel)
    green_pixel = torch.tensor([0, 1, 0], device=inputs.device, dtype=inputs.dtype).view(1, 3, 1, 1)
    is_green = torch.all(inputs == green_pixel, dim=1)
    start_positions = torch.nonzero(is_green, as_tuple=False)

    # Initialize neighbors count tensor
    neighbors = torch.zeros(batch_size, device=inputs.device)

    # Neighbor offsets to check adjacent nodes (2 units away in each direction)
    neighbor_offsets = torch.tensor([[-2, 0], [2, 0], [0, -2], [0, 2]], device=inputs.device)

    # Loop through each offset and count neighbors
    for dx, dy in neighbor_offsets:
        neighbor_positions = start_positions[:, 1:] + torch.tensor([dx, dy], device=inputs.device)
        
        # Filter out invalid positions
        valid_mask = (neighbor_positions[:, 0] >= 0) & (neighbor_positions[:, 0] < pixels) & \
                     (neighbor_positions[:, 1] >= 0) & (neighbor_positions[:, 1] < pixels)
        valid_positions = neighbor_positions[valid_mask]
        batch_indices = start_positions[valid_mask, 0]

        # Increment neighbor counts for valid positions
        neighbors.scatter_add_(0, batch_indices, 
                               inputs[batch_indices, :, valid_positions[:, 0], valid_positions[:, 1]].any(dim=1).float())

    # Adjust count to correct for over-counting
    return neighbors.cpu().numpy() / 4

def get_mmd(X, Y):
    """ Compute the Maximum Mean Discrepancy (MMD) between two sets of samples using parallel PyTorch operations."""

    n = X.shape[0]
    m = Y.shape[0]

    # Compute the pairwise distance matrices
    XX = torch.cdist(X, X, p=2)
    YY = torch.cdist(Y, Y, p=2)
    XY = torch.cdist(X, Y, p=2)

    # Compute the kernel values
    K_XX = torch.exp(-0.5 * XX ** 2)
    K_YY = torch.exp(-0.5 * YY ** 2)
    K_XY = torch.exp(-0.5 * XY ** 2)

    # Compute the MMD terms
    term_1 = K_XX.sum() / (n ** 2)
    term_2 = K_YY.sum() / (m ** 2)
    term_3 = -2 * K_XY.sum() / (n * m)

    return term_1 + term_2 + term_3, term_1, term_2, term_3