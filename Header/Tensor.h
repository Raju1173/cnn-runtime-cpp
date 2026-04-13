#pragma once
#include<vector>
#include<cstddef>

struct Tensor 
{
	float* pData;
	std::vector<size_t> shape;
	size_t numel;

	Tensor(const std::vector<size_t>& shape);

	void RELU();

	void zeros();

	void fillRandom();

	void print();

	~Tensor();
};

struct MaxPoolCache
{
	std::vector<size_t> indices;
};

Tensor add(const Tensor& a, const Tensor& b);

Tensor im2col(const Tensor& input, size_t R, size_t S);

Tensor GEMM(const Tensor& a, const Tensor& b);

Tensor reshape(const Tensor& input, const std::vector<size_t>& newShape);

Tensor conv2DForward(const Tensor& input, const Tensor& weights, const Tensor& bias);

Tensor MaxPool(const Tensor& input, MaxPoolCache& cache);