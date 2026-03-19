#pragma once
#include<vector>
#include<cstddef>

struct View2D
{
	float* data;
	size_t rows;
	size_t cols;
	ptrdiff_t stride_row;
	ptrdiff_t stride_col;
};

struct Tensor 
{
	float* pData;
	std::vector<size_t> shape;
	size_t numel;

	Tensor(const std::vector<size_t>& shape);

	Tensor im2col(size_t R, size_t S) const;

	void zeros();

	void fillRandom();

	void print();

	~Tensor();
};

void add(const Tensor& a, const Tensor& b, Tensor& out);

void GEMM(const Tensor& a, const Tensor& b, Tensor& out);

Tensor reshape(const Tensor& input, const std::vector<size_t>& newShape);

Tensor conv2DForward(const Tensor& input, const Tensor& weights, const Tensor& bias);
