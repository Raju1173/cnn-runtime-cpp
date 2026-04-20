#pragma once
#include "Tensor.h"

void Add(const Tensor& A, const Tensor& B, Tensor& out);

void Im2col(const Tensor& input, size_t R, size_t S, Tensor& out);

void Col2im(const Tensor& grad_col, size_t C, size_t H, size_t W, size_t R, size_t S, Tensor& grad_input);

void GEMM(const Tensor& A, const Tensor& B, Tensor& out);

void SGD(Tensor& param, const Tensor& grad, float lr);