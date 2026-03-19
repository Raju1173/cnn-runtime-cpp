# cnn-runtime-cpp

A minimal convolutional neural network runtime built from scratch in C++.

## Motivation

This project is an attempt to rebuild a small CNN runtime from first principles, with a focus on:

- understanding data layout and memory access
- implementing core numerical kernels manually
- exploring low level performance optimizations

## What’s Implemented

### Tensors
- Manual memory management (raw pointers)
- Explicit shape tracking
- Contiguous storage (row major layout)

### Numerical Operations
- Element-wise addition
- General Matrix multiplication (GEMM)

### Optimized GEMM
- Loop reordering for cache locality
- Reduced indexing overhead
- Compiler auto-vectorization
- Cache blocking (tiling)

Achieved ~25 GFLOPS on 512×512 matrices on a single core.

### Layers & Transformations

- im2col
- Reshape
- 2D Convolution (im2col + GEMM).

## GEMM Benchmarks (512×512)

| Version            | Time (ms) |
|--------------------|-----------|
| Naive (ijk)        | ~66 ms    |
| Reordered (ikj)    | ~43 ms    |
| + pointer reuse    | ~13 ms    |
| + blocking         | ~10.4 ms  |

CPU : Intel(R) Core(TM) i5-13420H (13th Gen)
