#include "Ops.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape), pData(nullptr), numel(1)
{
	for (size_t dim : shape)
	{
		numel *= dim;
	}

	pData = (float*)std::malloc(numel * sizeof(float));
}

Tensor::~Tensor()
{
	free(pData);
}

Tensor Conv2DForward(const Tensor& input, const Tensor& weights, const Tensor& bias)
{
	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t K = weights.shape[0];
	size_t C = weights.shape[1];
	size_t R = weights.shape[2];
	size_t S = weights.shape[3];

	size_t H_out = H - R + 1;
	size_t W_out = W - S + 1;

	Tensor col({ C * R * S, H_out * W_out });

	Im2col(input, R, S, col);

	Tensor w_flat = Reshape(weights, { K, C * R * S });

	Tensor out({ K, H_out * W_out });

	GEMM(w_flat, col, out);

	for (size_t k = 0; k < K; k++)
	{
		float B = bias.pData[k];
		float* row = out.pData + k * (H_out * W_out);

		for (size_t i = 0; i < H_out * W_out; i++)
		{
			row[i] += B;
		}
	}

	return Reshape(out, { K, H_out, W_out });
}

void Tensor::zeros()
{
	for (size_t i = 0; i < numel; i++)
	{
		pData[i] = 0;
	}
}

void Tensor::fillRandom()
{
	for (size_t i = 0; i < numel; i++)
	{
		pData[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
	}
}

std::ostream& operator << (std::ostream& os, const Tensor& T)
{
	if (T.shape.size() == 2)
	{
		size_t rows = T.shape[0];
		size_t cols = T.shape[1];

		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				os << T.pData[i * cols + j] << " ";
			}

			os << "\n";
		}
	}

	else
	{
		for (size_t i = 0; i < T.numel; i++)
		{
			os << T.pData[i] << " ";
		}

		os << "\n";
	}

	return os;
}
