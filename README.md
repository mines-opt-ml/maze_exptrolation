# Maze Extrapolation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

*Maze Extrapolation* is a research project dedicated to experimenting with neural networks designed to extrapolate solutions in maze-solving tasks. By leveraging state-of-the-art deep learning architectures, the project aims to predict and extend paths in complex mazes beyond the patterns encountered during training. This work explores the capabilities and limitations of neural networks in extrapolative reasoning, contributing to advancements in artificial intelligence and machine learning.

Read our paper [here](INSERT_UPDATED_LINK_HERE).

The BibTeX citation is:

    @article{fung2023mazeextrapolation,
      title={Maze Extrapolation: Neural Networks and Path Prediction},
      author={Fung, S W and Doe, J},
      journal={arXiv preprint arXiv:1234.56789},
      year={2023},
      url={https://arxiv.org/abs/1234.56789},
    }

## Features
- Extrapolation of paths in mazes using deep learning.
- Built-in support for [dt_net](https://github.com/aks2203/deep-thinking).
- Modular architecture for easy integration of new models.

## Installation
1. Clone this repository:
    ```bash
    git clone git@github.com:mines-opt-ml/maze_extrapolation.git
    ```
2. Navigate to the project directory:
    ```bash
    cd maze_extrapolation
    ```
3. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. (Optional) To deactivate the virtual environment when you're done:
    ```bash
    deactivate
    ```

## Usage
To run the models, use the following command:
```bash
python main.py --model dt_net
