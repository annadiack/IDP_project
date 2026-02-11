# Effect of Static Weight Sparsity in Neural Networks  
### Deep Learning Project – University of Basel (Fall 2025)

---

## Overview

This project investigates the effect of **static weight sparsity** on the performance and training dynamics of neural networks.

Modern neural networks are often heavily overparameterized, containing far more weights than seemingly necessary for strong predictive performance. This raises a natural question:

> How many parameters does a neural network actually need?

To study this, we systematically introduce **static sparsity** into a multi-layer perceptron (MLP) trained on the Fashion-MNIST dataset. A fixed fraction of weights is set to zero *before training*, and this sparsity pattern remains unchanged throughout optimization.

Our goal is not to design the most efficient sparse architecture, but rather to empirically analyze how increasing sparsity affects:

- Test accuracy  
- Training dynamics  
- Convergence behavior  
- Gradient-related signals  

The project is part of the *Foundations in Deep Learning* course and is implemented in **PyTorch**.

---

## Research Questions

1. How does increasing static weight sparsity affect classification accuracy?
2. How robust is an MLP to moderate levels of sparsity?
3. At what sparsity threshold does performance significantly degrade?
4. How does sparsity influence optimization dynamics during training?

---

## Methodology

### Model

- Multi-Layer Perceptron (MLP)
- Fully connected architecture
- ReLU activations
- Softmax output layer

### Dataset

- **Fashion-MNIST**
- 10-class image classification task
- Grayscale images (28×28)

### Sparsity Setup

- Random sparsity masks applied to weight matrices
- Fixed sparsity levels across experiments (e.g., 0% to high sparsity)
- Mask remains constant during training
- No re-growth or dynamic pruning

This setup allows us to isolate the effect of reduced parameter capacity without introducing additional structural learning mechanisms.

---

## Experimental Design

For each sparsity level, we:

- Train the network with identical hyperparameters
- Track training and validation loss
- Evaluate final test accuracy
- Analyze optimization-related quantities (e.g., gradient norms)

To ensure consistency, all experiments use fixed seeds and controlled training settings.

---

## Repository Structure

