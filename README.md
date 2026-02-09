# Effect of Sparsity in Weight Matrices in Neural Networks
### Deep Learning Project – University of Basel (Fall 2025)

---

## Overview

<<<<<<< Local Changes
This project investigates **iteartive pruning** as a method to introduce sparsity in neural networks and analyzes it effects on training dynamics, performance, and efficiency. We study how gradually removing weights druing training impacts convergence, accuracy and computational cost.
Experiments are conducted on multilayer perceptons (NLPs) and convelutional neural networks (CNNs) using pyTorch.

- Test accuracy
- Training and validation loss
- Convergence speed
- Training time per epoch
- Memory usage and effective parameter count


The project is part of the *Foundations in Deep Learning* course and is implemented in **PyTorch**.

---

## Research Questions

1. How does iterative pruning affect training dynamics and convergence?
2. How much sparsity can be introduced without significantly degrading accuracy?
3. How robust is iterative pruning across different model configurations?
4. What are the trade-offs between accuracy, training time, and memory usage?
5. Does iterative pruning behave differently in MLPs compared to CNNs?


---

## Repository Structure

---

.
├── src/.              # MLP and CNN architectures
├── pruning/           # Iterative pruning methods
├── experiments/       # Experiment configurations and scripts
├── results/           # Logged metrics and plots
├── utils/             # Training and evaluation utilities
└── README.md


## Project Checklist


### Implementation
- [x] Dense baseline implementation
- [x] Iterative pruning (magnitude-based) implementation
- [x] Configurable pruning schedule
- [x] Support for MLP architectures
- [x] Support for CNN architectures

### Experiments
- [x] Dense vs iterative pruning comparison
- [x] Experiments across different sparsity levels
- [x] Experiments across different model configurations (depth / width)
- [x] Multiple runs for stability analysis (fixed seeds)
- [x] Logging of sparsity level during training

### Evaluation & Analysis
- [x] Test accuracy evaluation
- [x] Training and validation loss tracking
- [x] Convergence speed analysis
- [x] Training time measurement
- [x] Memory usage / effective parameter count analysis

### Reproducibility & Documentation
- [x] Reproducible experiment setup
- [x] Clear experiment configuration files
- [x] Well-documented codebase
- [x] Clean repository structure
- [x] README documentation

