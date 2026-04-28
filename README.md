# Minimal CNN runtime

A small CNN runtime built from scratch in C++, including both training and real-time inference.

## Overview

This project rebuilds a complete CNN pipeline from first principles, focusing on :

- low level memory control
- cache efficient numerical kernels
- explicit forward and backward propagation
- end to end training without external ML libraries

## Features
### Tensor Core
- Manual memory management using raw pointers
- Explicit shape tracking
- Contiguous row major layout
- Zero overhead indexing
### Optimized GEMM
- Loop reordering (ikj) for cache locality
- Pointer reuse to eliminate redundant indexing
- Compiler auto-vectorization
- Cache blocking (tiling)
### Performance:
~25 GFLOPS on 512×512 matrix multiplication (single core)

## Neural Network Support
### Layers
- Convolution (im2col + GEMM)
- ReLU
- Max Pooling
- Fully Connected (Linear)
### Training
- Full backward propagation for all layers
- Softmax + Cross Entropy loss
- Stochastic Gradient Descent (SGD)
### Data Flow
- im2col / col2im transformations
- Manual gradient propagation across layers

## Results
 (Training\DoodlePredictor.py)

## Benchmarks (GEMM 512×512)
| Version            | Time (ms) |
|--------------------|-----------|
| Naive (ijk)        | ~66 ms    |
| Reordered (ikj)    | ~43 ms    |
| + pointer reuse    | ~13 ms    |
| + blocking(64 X 64)| ~10.4 ms  |

CPU: Intel i5-13420H (13th Gen)

## Demo (Training\DoodlePredictor.py)

- Trained on a subset of Google QuickDraw (10 classes)
- Achieved ~70% accuracy on test data
- Supports real time inference on hand-drawn input

## Future Work
- Inference latency optimization
- Multithreaded execution

## Build
```
mkdir build
cd build
cmake ..
make
```
