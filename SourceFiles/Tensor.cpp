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

Tensor Add(const Tensor& A, const Tensor& B)
{
	if (A.numel != B.numel)
		throw std::runtime_error("add : numel mismatch");

	Tensor out(A.shape);

	for (size_t i = 0; i < A.numel; i++)
	{
		out.pData[i] = A.pData[i] + B.pData[i];
	}
}

Tensor GEMM(const Tensor& A, const Tensor& B)
{
	if (A.shape.size() != 2 || B.shape.size() != 2)
		throw std::runtime_error("GEMM : only 2D tensors supported");

	size_t m = A.shape[0];
	size_t kA = A.shape[1];
	size_t kB = B.shape[0];
	size_t n = B.shape[1];

	Tensor out({ m, n });

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

	return out;
}

Tensor Im2col(const Tensor& input, size_t R, size_t S)
{
	size_t C = input.shape[0];
	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t H_out = H - R + 1;
	size_t W_out = W - S + 1;

	size_t outRows = C * R * S;
	size_t outCols = H_out * W_out;

	Tensor out({ outRows, outCols });

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

	return out;
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

	Tensor col = Im2col(input, R, S);
	Tensor w_flat = Reshape(weights, { K, C * R * S });

	Tensor out({ K, H_out * W_out });

	 out = GEMM(w_flat, col);

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

void Tensor::RELU()
{
	for (size_t i = 0; i < numel; i++)
	{
		pData[i] = std::max(pData[i], 0.0f);
	}
}

Tensor MaxPool(const Tensor& input, MaxPoolCache& cache)
{
	size_t C = input.shape[0];
	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t H_out = H / 2;
	size_t W_out = W / 2;

	Tensor out({ C, H_out, W_out });

	cache.indices.resize(C * H_out * W_out);

	for (size_t c = 0; c < C; c++)
	{
		for (size_t h_out = 0; h_out < H_out; h_out++)
		{
			for (size_t w_out = 0; w_out < W_out; w_out++)
			{
				size_t in_row = h_out * 2;
				size_t in_col = w_out * 2;

				float maxVal = input.pData[c * H * W + in_row * W + in_col];
				size_t maxIndex = c * H * W + in_row * W + in_col;

				for (int r = 0; r < 2; r++)
				{
					for (int s = 0; s < 2; s++)
					{
						float val = input.pData[c * H * W + (in_row + r) * W + (in_col + s)];

						if (val > maxVal)
						{
							maxVal = val;
							maxIndex = c * H * W + (in_row + r) * W + (in_col + s);
						}
					}
				}

				size_t outIdx = c * H_out * W_out + h_out * W_out + w_out;

				out.pData[outIdx] = maxVal;
				cache.indices[outIdx] = maxIndex;
			}
		}
	}

	return out;
}

Tensor Reshape(const Tensor& input, const std::vector<size_t>& newShape)
{
	size_t newNumel = 1;

	for (size_t dim : newShape)
	{
		newNumel *= dim;
	}

	if (newNumel != input.numel)
		throw std::runtime_error("reshape : numel mismatch");

	Tensor out(newShape);

	std::copy(input.pData, input.pData + input.numel, out.pData);

	return out;
}

Tensor Linear(const Tensor& input, const Tensor& weights, const Tensor& bias)
{
	if (input.shape.size() != 1 || weights.shape.size() != 2 || bias.shape.size() != 1)
		throw std::runtime_error("Linear : only 1D input, 2D weights and 1D bias supported");

	size_t inFeatures = input.shape[0];
	size_t outFeatures = weights.shape[0];

	if (weights.shape[1] != inFeatures || bias.shape[0] != outFeatures)
		throw std::runtime_error("Linear : shape mismatch");

	Tensor out({ outFeatures });

	out = Reshape(GEMM(weights, Reshape(input, { inFeatures, 1 })), {outFeatures});

	out = Add(out, bias);

	return out;
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

std::ostream& operator<<(std::ostream& os, const Tensor& T)
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
