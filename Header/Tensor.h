#pragma once
#include<vector>
#include<cstddef>
#include<iostream>

class Tensor
{
public :
	float* pData;
	std::vector<size_t> shape;
	std::vector<size_t> strides;
	size_t numel;

	Tensor(const std::vector<size_t>& shape);

	void zeros();

	void fillRandom();

	~Tensor();
};

std::ostream& operator << (std::ostream& os, const Tensor& T);

Tensor Add(const Tensor& A, const Tensor& B);

Tensor Im2col(const Tensor& input, size_t R, size_t S);

Tensor GEMM(const Tensor& A, const Tensor& B);

Tensor Conv2DForward(const Tensor& input, const Tensor& weights, const Tensor& bias);

void RELU(const Tensor& input);

struct MaxPoolCache // This is not supposed to be here but just for now for convenience...
{
	std::vector<size_t> indices;
};

Tensor MaxPool(const Tensor& input, MaxPoolCache& cache);

Tensor Reshape(const Tensor& input, const std::vector<size_t>& newShape); // This should not be copying memory. So basically, implement strides in the future...

Tensor Linear(const Tensor& input, const Tensor& weights, const Tensor& bias);