import numpy as np
import pickle

from src.utils.testing import Test

for model_name in ['dt_net', 'pi_net']:
    test = Test(
        model_name=model_name,
        dataset='maze-dataset',
        num_mazes=100,
        iters=[30, 100, 300],# 1000, 3000],
        maze_sizes=[9],# 19, 29, 39, 49, 59, 69, 79, 89, 99],
        percolations=[0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04],
        deadend_start=True,
        batch_size=1)
    
    test.get_accuracies()

    # Save the Test object
    file_name = f'outputs/extrapolation/tests/{test.get_name()}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(test, f)
