import torch
from easy_to_hard_data import MazeDataset as EasyToHardMazeDataset
from maze_dataset import SolvedMaze, LatticeMaze, set_serialize_minimal_threshold
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.dataset.rasterized import MazeDatasetConfig, MazeDataset, RasterizedMazeDatasetConfig, RasterizedMazeDataset
set_serialize_minimal_threshold(int(10**7)) # set this threshold to prevent crashing on large datasets. Will be fixed soon.
import yaml
from omegaconf import OmegaConf

import src.models as models
from src.utils.device import get_device
from src.models.dt_net import DTNet
from src.models.it_net import ITNet
from src.models.pi_net import PINet

def load_model(model_name, verbose=True):
    """Load the saved model onto device."""

    # Get device
    device = get_device(verbose=True)

    # Initialize model and load weights
    model = None
    state_dict = None
    if model_name == 'dt_net':
        model = DTNet()     
        state_dict = torch.load('models/dt_net.pth', map_location=device, weights_only=True)['net']

    elif model_name == 'it_net':
        model = ITNet()
        state_dict = torch.load('models/it_net.pth', map_location=device, weights_only=True)['net']

    elif model_name == 'pi_net':
        cfg_path = 'models/pi_net/aric/config.yaml'
        model_path = 'models/pi_net/aric/model_best_130_100.0.pth'

        # Get config dictionary, convert to omega config, and fix attributes
        with open(cfg_path) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg_dict)
        cfg.problem.deq.jacobian_factor = 1.0
        cfg.problem.model.model_path = model_path

        # Create model and load weights
        model = PINet(width=cfg.problem.model.width, in_channels=3, config=cfg)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)['net']

    # Fix state_dict keys
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load weights to model
    model.to(device)
    model.eval()
    model.load_state_dict(new_state_dict, strict=True)

    if verbose:
        print(f'Loaded {model_name} to {device}')

    return model

def get_mazes(dataset='maze-dataset', maze_size=9, num_mazes=10, gen='dfs_perc', percolation=0.0, deadend_start=True):
    """ Generate mazes of the given size and number, 
        from the given dataset, and load to device"""
    
    if dataset == 'maze-dataset':
        """ https://github.com/understanding-search/maze-dataset """

        assert maze_size % 2 == 1
        grid_n = (maze_size + 1) // 2
        
        # Generate base maze dataset
        if gen == 'dfs':
            maze_ctor = LatticeMazeGenerators.gen_dfs
            maze_ctor_kwargs = dict()
        elif gen == 'dfs_perc':
            maze_ctor = LatticeMazeGenerators.gen_dfs_percolation
            maze_ctor_kwargs = dict(p=percolation)
        elif gen == 'percolation':
            maze_ctor = LatticeMazeGenerators.gen_percolation
            maze_ctor_kwargs = dict(p=percolation)
        endpoint_kwargs=dict(deadend_start=deadend_start, endpoints_not_equal=True)

        base_dataset = MazeDataset.from_config(
            MazeDatasetConfig(
                name='test',
                grid_n=grid_n,
                n_mazes=num_mazes,
                seed=42,
                maze_ctor=maze_ctor, 
                maze_ctor_kwargs=maze_ctor_kwargs,
                endpoint_kwargs=endpoint_kwargs
            ),
            local_base_path='data/maze-dataset/',
        )

        # Generate rasterized maze dataset
        dataset = RasterizedMazeDataset.from_base_MazeDataset(
            base_dataset=base_dataset,
            added_params=dict(
                remove_isolated_cells=True,
                extend_pixels=True, # maps from 1x1 to 2x2 pixels and adds 3 padding
            )
        )

        dataset = dataset.get_batch(idxs=None)

        # Get inputs
        inputs = dataset[0,:,:,:]
        inputs = inputs / 255.0
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs.float().detach().to(get_device(), dtype=torch.float32)

        # Get solutions
        solutions = dataset[1,:,:, :]
        solutions = solutions / 255.0
        solutions = solutions.permute(0, 3, 1, 2)
        solutions, _ = torch.max(solutions, dim=1)
        solutions = solutions.float().detach().to(get_device(), dtype=torch.float32) 

    elif dataset == 'easy-to-hard-data':
        """ https://github.com/aks2203/easy-to-hard-data """

        assert maze_size in [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 59]
        # 50,000 training mazes for maze_size [9]
        # 10,000 testing mazes for each smaller maze_size in [9, 11, 13, 15, 17]
        # 1,000 testing mazes for each larger maze_size in [19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 59]

        maze_dataset = EasyToHardMazeDataset(root='data/easy-to-hard-data/', train=False, size=maze_size)
        inputs = maze_dataset.inputs[:num_mazes].float().detach().to(get_device(), dtype=torch.float32)
        solutions = maze_dataset.targets[:num_mazes].float().detach().to(get_device(), dtype=torch.float32)

    return inputs, solutions