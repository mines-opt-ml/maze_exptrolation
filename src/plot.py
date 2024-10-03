import pickle

from src.utils.plotting import plot_extrapolation

# Specify file to load test from
model_name = 'dt_net'

# Maze size and deadend_start
deadend_start = False
file_name = f'outputs/extrapolation/tests/model-{model_name}_dataset-maze-dataset_num_mazes-100_iters-[30, 100, 300, 1000, 3000]_maze_sizes-[9, 19, 29, 39, 49, 59, 69, 79, 89, 99]_percolations-[0.0]_deadend_start-{deadend_start}.pkl'

## Percolation
#file_name = f'outputs/extrapolation/tests/model-{model_name}_dataset-maze-dataset_num_mazes-100_iters-[30, 100, 300]_maze_sizes-[9]_percolations-[0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]_deadend_start-True.pkl'

# Load the Test object and print accuracies
with open(file_name, 'rb') as f:
    test = pickle.load(f)
print()
print(test.get_name())
print()
test.print()

for i, iter in enumerate(test.iters):
    print(f'Accuracy at {iter} iterations: {test.accuracies[i]}')

#Plot the extrapolation
plot_extrapolation(test,
                   split_neighbors=True,
                   iter=3000,
                   font_size=30,
                   fig_size=(16,8),
                   file_name=f'outputs/extrapolation/plots/{model_name}_deadend_start-{deadend_start}_split_neighbors')