import numpy as np
import pickle

from src.utils.tda import Analysis

for model_name in ['dt_net']: 
    analysis = Analysis(
        maze_sizes=[9, 19, 29, 39, 49, 59, 69],
        num_mazes=100, 
        model_name=model_name, 
        iters=list(range(3001, 3401)), 
        dtype=np.float64,
        embed_dim=0, 
        delay=1,
        max_homo=1)

    analysis.analyze(verbose=True)