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

Read our paper: [On Logical Extrapolation for Mazes with Recurrent and Implicit Networks](https://arxiv.org/search/math?searchtype=author&query=Fung,+S+W).

The BibTeX citation is:

    @article{knutson_maze-extrapolation_2024,
      title = {On Logical Extrapolation for Mazes with Recurrent and Implicit Networks},
      author = {
        Brandon Knutson and 
        Amandin Chyba Rabeendran and 
        Michael Ivanitskiy and 
        Jordan Pettyjohn and 
        Cecilia Diniz-Behn and 
        Samy Wu Fung and 
        Daniel McKenzie
      },
      journal = {arXiv preprint arXiv:1234.56789},
      year = {2023},
      url = {https://arxiv.org/abs/1234.56789},
    }

## Features
- Extrapolation accuracy quantification for models trained on maze images to produce solution path images.
- Contains a model from [Bansal et. al.](https://github.com/aks2203/deep-thinking).
- Modular architecture for easy integration of new models, just inherit from [BaseNet](src/models/base_net.py.)

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
To evaluate the models:
   - Extrapolation accuracy testing:
     ```bash
     python -m src.test
     ```
   - Topological data analysis:
     ```bash
     python -m src.analyze

See [notebooks](/notebooks/) folder for demos.