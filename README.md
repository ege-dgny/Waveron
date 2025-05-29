# Waveron Networks for Edge Computing: A Comparative Study of Wavelet-Based Neural Networks for Robust Feature Extraction

This repository contains the PyTorch implementation for the research paper: "Waveron Networks for Edge Computing: A Comparative Study of Wavelet-Based Neural Networks for Robust Feature Extraction." The project introduces and evaluates the "Waveron Network," a novel architecture leveraging Discrete Wavelet Transforms (DWT) and learnable operations for efficient and robust image classification, particularly on noisy MNIST.

## Project Overview

The core motivation is to develop neural network architectures that are both parameter-efficient (suitable for edge computing) and robust to common image corruptions like Gaussian noise. This research compares several Waveron Network variants against:

- Standard Multi-Layer Perceptrons (MLPs)
- A Fourier-transform-based network (FourierNet)

Experiments focus on the MNIST dataset, with analyses of performance on clean and noisy data.

## Key Features

### Waveron Network Architectures:
- Implementations of multi-layer Waveron Networks using learnable 2D convolutions within Waveron modules.
- Variants include different aggregation strategies (summation vs. channel-wise concatenation) and depths (single vs. two-layer).
- (Previous iterations also explored mask-based Waverons, the current primary version uses convolutions).

### Baseline Models:
- Multi-Layer Perceptron (MLP).
- FourierNet (FFT + MLP backend).

### Experimental Framework (run_experiment.py):
- Train and evaluate specified models on various datasets (MNIST, noisy MNIST).
- Conduct robustness evaluations against varying levels of Gaussian noise.
- Support for comparing different wavelet families within the Waveron Network.

### Visualization Utilities (utils/plotting.py):
- Training performance (loss, accuracy).
- t-SNE projection of penultimate layer features.
- Waveron-specific: learned convolutional filters, intermediate feature maps, average sub-band energies.
- FourierNet-specific: average 1D radial power spectrum, average 2D cropped power spectrum.
- Combined plots for robustness comparison (Accuracy vs. Noise Level).

## Repository Structure

```
waveron_research_project/
├── networks/                   # Model definitions
│   ├── __init__.py
│   ├── waveron_network.py
│   ├── mlp_network.py
│   └── fourier_network.py
├── utils/                      # Helper utilities
│   ├── __init__.py
│   ├── custom_transforms.py    # For adding noise, etc.
│   ├── dataset_loader.py
│   └── plotting.py
├── results/                    # Directory where output plots and data will be saved
├── main.py                     # Main script to run experiments
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- PyWavelets (`pip install PyWavelets`)
- scikit-learn (`pip install scikit-learn`)
- Matplotlib (`pip install matplotlib`)
- NumPy (`pip install numpy`)

Install dependencies using pip:

```bash
pip install torch torchvision torchaudio pytorch_wavelets scikit-learn matplotlib numpy
```

## Running Experiments

The main script to run experiments is `main.py`. It uses argparse for configuration.

**Example: Train and evaluate Waveron (summation, db4 wavelet) on noisy MNIST and test robustness to various noise levels:**

```bash
python main.py \
    --model_types waveron \
    --dataset mnist_noisy \
    --noise_std 0.2 \
    --wavelet db4 \
    --kernel_size 3 \
    --waveron_channels 8 \
    --epochs 10 \
    --eval_noise_levels 0.0 0.1 0.2 0.3 0.4 0.5
```

**Example: Compare Waveron, MLP, and FourierNet trained on clean MNIST, evaluated on noisy MNIST:**

```bash
python main.py \
    --model_types waveron mlp fourier \
    --dataset mnist \
    --eval_noise_levels 0.0 0.1 0.2 0.3 0.4 0.5 \
    --epochs 10 \
    --wavelet db4 --kernel_size 3 --waveron_channels 8 \
    --mlp_hidden_dims 512 256 \
    --fourier_crop_fraction 0.5 --fourier_mlp_hidden_dims 256 128
```

**Example: Evaluate robustness of different wavelets for the waveron_concat model:**

```bash
python main.py \
    --model_types waveron_concat \
    --eval_wavelets_robustness \
    --eval_noise_levels 0.0 0.1 0.2 0.3 0.4 0.5 \
    --kernel_size 3 \
    --waveron_channels 8 \
    --epochs 10
```

Refer to `main.py` for all available command-line arguments and their defaults. Experimental results (plots, etc.) will be saved in the `results/` directory, organized by experiment configuration and timestamp.

## License

[Specify your chosen license, e.g., MIT, Apache 2.0, or leave as TBD] 
