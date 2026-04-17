#include "Ops.h"
#include <stdexcept>

void Add(const Tensor& a, const Tensor& b, Tensor& out)
{
	if (a.numel != b.numel || a.numel != out.numel)
		throw std::runtime_error("add : numel mismatch");

	for (size_t i = 0; i < a.numel; i++)
	{
		out.pData[i] = a.pData[i] + b.pData[i];
	}
}

void Im2col(const Tensor& input, size_t R, size_t S, Tensor& out)
{
	size_t C = input.shape[0];
	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t H_out = H - R + 1;
	size_t W_out = W - S + 1;

	size_t outRows = C * R * S;
	size_t outCols = H_out * W_out;

	float* outData = out.pData;

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

						outData[row * outCols + col] = input.pData[c * H * W + in_row * W + in_col];
					}
				}
			}
		}
	}
}

void GEMM(const Tensor& A, const Tensor& B, Tensor& out)
{
	if (A.shape.size() != 2 || B.shape.size() != 2)
		throw std::runtime_error("GEMM : only 2D tensors supported");

	size_t m = A.shape[0];
	size_t kA = A.shape[1];
	size_t kB = B.shape[0];
	size_t n = B.shape[1];

	if (out.shape[0] != m || out.shape[1] != n)
		throw std::runtime_error("GEMM : output shape mismatch");

	if (kA != kB)
		throw std::runtime_error("GEMM : shape mismatch");

	out.zeros();

	//Blocked + Reordered GEMM :

	int BlockSize = 64;

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
						float aik = A.pData[i * kA + k];
						float* bRow = B.pData + k * n;

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
			float aik = A.pData[i * kA + k];
			float* bRow = B.pData + k * n;

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
				sum += A.pData[i * kA + k] * B.pData[k * n + j];
			}

			out.pData[i * n + j] = sum;
		}
	}
	*/
}