#pragma once
#include "Tensor.h"

void Add(const Tensor& A, const Tensor& B, Tensor& out);

void Im2col(const Tensor& input, size_t R, size_t S, Tensor& out);

void GEMM(const Tensor& A, const Tensor& B, Tensor& out);
