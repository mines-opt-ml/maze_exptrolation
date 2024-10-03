import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
from IPython.display import Image, IFrame, HTML

from src.utils.testing import Test
from src.utils.tda import get_max_death
from src.utils.pca import get_pca
from src.utils.config import default_config

def plot_mazes(inputs, predictions=None, solutions=None, unmasked_predictions=None, font_size=12, file_name=None):
    """
    Plots the maze inputs, predicted solutions, and real solutions.
    If unmasked_predictions are provided, also plots the unmasked predicted solutions.
    If file_name is given, save figure to file.
    """

    # Create a figure with available inputs, predictions, solutions, and unmasked_predictions in each row
    num_mazes = inputs.size(0)
    mazes_list = [inputs, unmasked_predictions, predictions, solutions]
    titles = ['Input', 'Unmasked Prediction', 'Prediction', 'Solution']

    # Determine the number of subplots
    num_subplots = sum(1 for mazes in mazes_list if mazes is not None)
    fig, axs = plt.subplots(num_mazes, num_subplots, figsize=(5*num_subplots, 6*num_mazes))

    # Ensure axs is always a 2D array
    axs = np.atleast_2d(axs)

    for i in range(num_mazes):
        j = 0
        for mazes, title in zip(mazes_list, titles):
            if mazes is not None:
                data = mazes[i].cpu().numpy()
                if title == 'Input':
                    data = data.transpose(1, 2, 0)
                axs[i, j].imshow(data.squeeze(), vmin=0, vmax=1, cmap='gray' if title != 'Input' else None)
                axs[i, j].set_title(title, fontsize=font_size, fontweight='bold')
                axs[i, j].axis('off')
                j += 1

    # Format figure
    plt.tight_layout()

    # Save or return figure
    if file_name is not None:
        plt.savefig(f'{file_name}.pdf', format='pdf')
    
    return fig

def plot_extrapolation(test, split_neighbors=False, iter=None, fig_size=(8,6), font_size=12, file_name=None):
    """
    Plot accuracy against extrapolation parameter (maze size or percolation) for different numbers of iterations.
    """

    if not split_neighbors:
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=fig_size)
        
        # For each number of iterations, plot accuracy against maze size 
        for i, iter in enumerate(test.iters):
            ax.plot(test.extrap_param, 
                    test.accuracies[i], 
                    marker='o', 
                    markersize=15,
                    linewidth=6,
                    label=f'{iter} iters', 
                    clip_on=False, 
                    zorder=5)

        ax.set_xlabel(test.extrap_param_name.capitalize(), fontsize=font_size+8, fontweight='bold')
        ax.set_xlim([0, max(test.extrap_param)+(0 if test.extrap_param_name == 'percolation' else 1)])
        ax.set_ylabel('Accuracy', fontsize=font_size+8, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=font_size, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(test.iters), prop=FontProperties(weight='bold', size=font_size))

        # Set the desired number of ticks automatically using MaxNLocator
        desired_num_ticks = 5  # Specify the number of ticks you want
        ax.xaxis.set_major_locator(MaxNLocator(nbins=desired_num_ticks))

        # Set font size and bold font weight for tick labels
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(font_size)
            label.set_fontweight('bold')
        ax.grid(True)
        plt.tight_layout()

    else:
        assert iter in test.iters, f'iter must be in {test.iters}'

        corrects = test.corrects[test.iters.index(iter)]
        
        # Constants
        neighbor_values = [1, 2, 3, 4]

        # Compute probability of maze with each neighbor value
        neighbor_prob = np.zeros((len(test.extrap_param), len(neighbor_values)))
        for i, n in enumerate(neighbor_values):
            neighbor_prob[:,i] = np.sum((test.start_neighbors == n), axis=1) / test.num_mazes

        # Compute probability of maze with each neighbor value and correct prediction
        neighbor_prob_correct = np.zeros((len(test.extrap_param), len(neighbor_values)))
        for i, n in enumerate(neighbor_values):
            neighbor_prob_correct[:,i] = np.sum((test.start_neighbors == n) * corrects, axis=1) / test.num_mazes

        # Create a figure and axis objects
        fig, axes = plt.subplots(1, neighbor_prob.shape[0], figsize=fig_size, sharey=True)

        for i, ax in enumerate(axes):
            # Plot attempted
            bar1 = ax.bar(neighbor_values, neighbor_prob[i], color='red', label='Incorrect' if i == 0 else "")
            # Plot correct
            bar2 = ax.bar(neighbor_values, neighbor_prob_correct[i], color='green', label='Correct' if i == 0 else "")

            ax.set_title(f'{test.extrap_param[i]}', fontsize=font_size)
            ax.set_xticks(neighbor_values)
            ax.set_ylabel('Probability' if i == 0 else '', fontsize=font_size+4, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=font_size)

        # Add a common title across the bottom
        fig.text(0.5, 0.03, 'Start Neighbors', ha='center', va='center', fontsize=font_size+4, fontweight='bold')
        # Add a common title across the top
        fig.text(0.5, 0.96, 'Maze Size', ha='center', va='center', fontsize=font_size+4, fontweight='bold')

        ## Add legend
        # plt.subplots_adjust(right=0.8)  # Adjust this value to make enough space for the legend
        # fig.legend([bar1, bar2], ['Incorrect', 'Correct'], bbox_to_anchor=(1, 0.5), fontsize=font_size+4)

    if file_name is not None:
        plt.savefig(f'{file_name}.pdf', format='pdf')
    
    return fig

def plot_diagram(D, threshold=None, font_size=12, file_name=None):
    """ Plot the persistence diagram D, ignoring infinity"""

    fig = plt.figure(figsize=(8, 8))

    # Plot (birth, death) pairs for each homology class
    for i in range(len(D)):
        plt.scatter(D[i][:, 0], D[i][:, 1], label=f'H{i}', s=15, clip_on=False)

    # Plot birth=death line
    max_death = get_max_death(D)
    plt.plot([0, 1.05*max_death], [0, 1.05*max_death], 'k--')

    if threshold is not None:
        # Plot persistence threshold line
        plt.plot([0, 1.05*max_death], [threshold*max_death, threshold*max_death + 1.05*max_death], '--', color='red')

    plt.xlabel('Birth', fontsize=font_size)
    plt.xlim(0, 1.05*max_death)
    plt.ylabel('Death', fontsize=font_size)
    plt.ylim(0, 1.05*max_death)
    plt.tight_layout()
    plt.legend(fontsize=font_size)

    if file_name is not None:
        plt.savefig(file_name)

    return fig

def plot_residuals(X, start_idx=None, fig_size=(8,8), font_size=12, file_name=None):
    """ Plot norm of differences between consecutive rows of flattened X """
    
    # Compute residuals
    X = X.reshape(X.shape[0], -1)
    residuals = torch.norm(X[1:] - X[:-1], dim=1).cpu().numpy()
    print(f'{max(residuals) = }')

    # Plot residuals
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.plot(range(start_idx, start_idx+X.shape[0]-1), residuals, markersize=20)
    ax.set_xlabel('Iteration', fontsize=font_size, fontweight='bold')
    ax.set_ylabel('Residual', fontsize=font_size, fontweight='bold')
    #ax.set_ylim(bottom=0, top=1.05*max(residuals))
    ax.set_ylim(bottom=10.35, top=11.35)

    # Set the desired number of ticks automatically using MaxNLocator
    desired_num_ticks = 5  # Specify the number of ticks you want
    ax.xaxis.set_major_locator(MaxNLocator(nbins=desired_num_ticks))

    # Set font size and bold font weight for tick labels
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(font_size)
    plt.tight_layout()
    
    if file_name is not None:
        plt.savefig(f'{file_name}.pdf', format='pdf')

    return fig

def plot_latents(latents, file_name=None, animate=False, duration=10, skip_frames=1, point_size=10, fig_size=(8, 8), font_size=12):
    """ Plot the latents by projecting onto 3 principal components, optionally animating the plot """
    
    # Project data onto 3 principal components
    if isinstance(latents, torch.Tensor):
        latents = latents.cpu().numpy()
    latents = latents.astype(np.float64)
    X = get_pca(latents, n=3)
    print(f'{X[:5] = }')
    print(f'{X[-5:] = }')

    # Set up figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
        
    # Set title and labels with font size and bold font weight
    ax.set_title('PCA Projection of Latent Iterates', fontsize=font_size, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=font_size, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=font_size, fontweight='bold')
    ax.set_zlabel('PC3', fontsize=font_size, fontweight='bold')

    # Set font size and bold font weight for tick labels
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontsize(font_size)
        label.set_fontweight('bold')

    # Set axis limits
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_zlim(x_min, x_max)

    # Automatically limit the number of ticks to 5
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

    # Color points by iteration
    colors = np.arange(len(X))
    norm = plt.Normalize(colors.min(), colors.max())
    cmap = plt.get_cmap('plasma')

    # Plot without animation
    if not animate:
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40) #, c=colors, cmap=cmap, norm=norm) Don't color by iteration
        #fig.colorbar(scatter)
        if file_name:
            pdf_file = f'{file_name}.pdf'
            plt.savefig(pdf_file, format='pdf')
            return IFrame(pdf_file, width=800, height=600)
        else:
            return fig

    # Plot with animation
    else:
        scatter = ax.scatter([], [], [], c=[], cmap=cmap, norm=norm)

        # Function to update the scatter plot during animation
        def update_plot(frame):
            print(f'Frame {frame+1}/{len(X)}', end='\r')
            scatter._offsets3d = (X[:frame, 0], X[:frame, 1], X[:frame, 2])
            scatter.set_array(colors[:frame])
            return scatter,

        # Reduce the number of frames by skipping some
        frames = range(0, len(X), skip_frames)
        interval = (duration * 1000) / len(frames)  # Adjust interval based on reduced frames

        ani = animation.FuncAnimation(fig, update_plot, frames=frames, interval=interval, blit=False)

        if file_name:
            gif_file = f'{file_name}.gif'
            ani.save(gif_file, writer='pillow', fps=1000/interval)
            plt.show()
            return Image(filename=gif_file)
        else:
            return HTML(ani.to_jshtml())