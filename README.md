# BPU: Biological Processing Unit

This repository implements a novel neural network architecture that incorporates the Drosophila larva connectome ([DOI: 10.1126/science.add9330](https://doi.org/10.1126/science.add9330)) as a recurrent neural network (RNN) layer and tests its performance on the MNIST dataset.

## Overview

We transform the complete neuronal wiring diagram of the Drosophila larva brain into a recurrent neural network weight matrix. This biological weight initialization method is compared against traditional artificial neural networks on image classification tasks.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Seaborn

## Project Structure

```
BPU/
├── data/                  # Connectome data and datasets
├── notebooks/            # Jupyter notebooks for training and analysis
├── plotting/             # Visualization scripts and figures
├── results/              # Saved model weights and experiment results
├── src/                  # Source code
│   ├── connectome.py    # Connectome processing utilities
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── net.py           # Neural network model definitions
│   └── utils.py         # Utility functions
└── config.yaml          # Configuration file for experiments
```

## Getting Started

See the [tutorial notebook](notebooks/tutorial.ipynb) for a step-by-step guide on configuring experiments, loading data, training models, and visualizing results.

## Citation

If you use this code or find this work helpful, please cite: