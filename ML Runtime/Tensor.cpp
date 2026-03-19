#include "Tensor.h"
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

inline void add(const Tensor& a, const Tensor& b, Tensor& out)
{
	if (a.numel != b.numel)
		throw std::runtime_error("add : numel mismatch");

	if (out.numel != a.numel)
		throw std::runtime_error("add : output size mismatch");

	for (size_t i = 0; i < a.numel; i++)
	{
		out.pData[i] = a.pData[i] + b.pData[i];
	}
}

void GEMM(const Tensor& a, const Tensor& b, Tensor& out)
{
	if (a.shape.size() != 2 || b.shape.size() != 2 || out.shape.size() != 2) 
		throw std::runtime_error("matmul : only 2D tensors supported");

	size_t m = a.shape[0];
	size_t kA = a.shape[1];
	size_t kB = b.shape[0];
	size_t n = b.shape[1];

	if (out.shape[0] != m || out.shape[1] != n)
		throw std::runtime_error("matmul : output shape mismatch");

	if (kA != kB)
		throw std::runtime_error("matmul : shape mismatch");

	out.zeros();

	//Blocked + Reordered GEMM :

	int BlockSize = 128;

	for (size_t ii = 0; ii < m; ii += BlockSize)
	{
		size_t i_max = std::min(ii + BlockSize, m);

		for (size_t kk = 0; kk < kA; kk += BlockSize)
		{
			size_t k_max = std::min(kk + BlockSize, kA);

			for (size_t jj = 0; jj < n; jj += BlockSize)
			{
				size_t j_max = std::min(jj + BlockSize, n);

				for (size_t i = ii; i < i_max; i++)
				{
					float* cRow = out.pData + i * n;

					for (size_t k = kk; k < k_max; k++)
					{
						float aik = a.pData[i * kA + k];
						float* bRow = b.pData + k * n;

						for (size_t j = jj; j < j_max; j++)
						{
							cRow[j] += aik * bRow[j];
						}
					}
				}
			}
		}
	}


	//Reordered GEMM :

	/*
	for (size_t i = 0; i < m; i++)
	{
		float* cRow = out.pData + i * n;

		for (size_t k = 0; k < kA; k++)
		{
			float aik = a.pData[i * kA + k];
			float* bRow = b.pData + k * n;

			for (size_t j = 0; j < n; j++)
			{
				cRow[j] += aik * bRow[j];
			}
		}
	}
	*/

	//Naive GEMM :

	/*
	for (size_t i = 0; i < m; i++) 
	{
		for (size_t j = 0; j < n; j++) 
		{
			float sum = 0.0f;

			for (size_t k = 0; k < kA; k++) 
			{
				sum += a.pData[i * kA + k] * b.pData[k * n + j];
			}
        
			out.pData[i * n + j] = sum;
		}
	}
	*/
}

Tensor Tensor::im2col(size_t R, size_t S)
{
	size_t C = shape[0];
	size_t H = shape[1];
	size_t W = shape[2];

	size_t H_out = H - R + 1;
	size_t W_out = W - S + 1;

	size_t out_rows = C * R * S;
	size_t out_cols = H_out * W_out;

	Tensor out({ out_rows, out_cols });

	float* out_data = out.pData;

	for (size_t c = 0; c < C; c++)
	{
		for (size_t r = 0; r < R; r++)
		{
			for (size_t s = 0; s < S; s++)
			{
				size_t row = c * R * S + r * S + s;

				for (size_t h_out = 0; h_out < H_out; h_out++)
				{
					for (size_t w_out = 0; w_out < W_out; w_out++)
					{
						size_t col = h_out * W_out + w_out;

						size_t in_row = h_out + r;
						size_t in_col = w_out + s;

						out_data[row * out_cols + col] = pData[c * H * W + in_row * W + in_col];
					}
				}
			}
		}
	}

	return out;
}

/*Tensor conv2DForward(const Tensor& input, const Tensor& weights, const Tensor& bias)
{
	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t K = weights.shape[0];
	size_t R = weights.shape[2];
	size_t S = weights.shape[3];

	size_t H_out = H - R + 1;
	size_t W_out = W - S + 1;

	Tensor col = input.im2col(R, S);
	Tensor w_flat = reshapeWeights(weights);

	Tensor out({ K, H_out * W_out });

	GEMM(w_flat, col, out);

	for (size_t k = 0; k < K; k++)
	{
		float b = bias.pData[k];
		float* row = out.pData + k * (H_out * W_out);

		for (size_t i = 0; i < H_out * W_out; i++)
		{
			row[i] += b;
		}
	}

	return reshape3D(out, K, H_out, W_out);
}*/

inline void Tensor::zeros()
{
	for (size_t i = 0; i < numel; i++)
	{
		pData[i] = 0;
	}
}

inline void Tensor::fillRandom()
{
	for (size_t i = 0; i < numel; i++)
	{
		pData[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
	}
}

void Tensor::print()
{
	if (shape.size() == 2)
	{
		size_t rows = shape[0];
		size_t cols = shape[1];

		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				std::cout << pData[i * cols + j] << " ";
			}
			std::cout << "\n";
		}
	}

	else
	{
		for (size_t i = 0; i < numel; i++)
		{
			std::cout << pData[i] << " ";
		}
		std::cout << "\n";
	}
}