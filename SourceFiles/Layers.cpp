#include "Layers.h"

void ReLU::forward(Tensor& input)
{
	for (size_t i = 0; i < input.numel; i++)
	{
		input.pData[i] = std::max(input.pData[i], 0.0f);
	}
}

Conv2D::Conv2D(size_t inChannels, size_t outChannels, size_t kernelSize) : weights({ outChannels, inChannels, kernelSize, kernelSize }), bias({ outChannels })
{
	weights.fillRandom();
	bias.fillRandom();
}

void Conv2D::forward(const Tensor& inp, Tensor& out)
{
	this->input = inp;

	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t K = weights.shape[0];
	size_t C = weights.shape[1];
	size_t R = weights.shape[2];
	size_t S = weights.shape[3];

	size_t H_out = H - R + 1;
	size_t W_out = W - S + 1;

	if (out.shape[0] != K || out.shape[1] != H_out || out.shape[2] != W_out)
		throw std::runtime_error("Conv2D: output shape mismatch");

	this->col = Tensor({ C * R * S, H_out * W_out });

	Im2col(input, R, S, this->col);

	weights.shape = { K, C * R * S };

	out.shape = { K, H_out * W_out };

	GEMM(weights, col, out);

	for (size_t k = 0; k < K; k++)
	{
		float B = bias.pData[k];
		float* row = out.pData + k * (H_out * W_out);

		for (size_t i = 0; i < H_out * W_out; i++)
		{
			row[i] += B;
		}
	}

	weights.shape = { K, C, R, S };

	out.shape = { K, H_out, W_out };
}

void MaxPool::forward(const Tensor& input, Tensor& out)
{
	size_t C = input.shape[0];
	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t H_out = H / 2;
	size_t W_out = W / 2;

	indices.resize(C * H_out * W_out);

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

				indices[outIdx] = maxIndex;
			}
		}
	}
}

Linear::Linear(size_t inFeatures, size_t outFeatures) : weights({ outFeatures, inFeatures }), bias({ outFeatures })
{
	weights.fillRandom();
	bias.fillRandom();
}

void Linear::forward(Tensor& inp, Tensor& out)
{
	this->input = inp;

	if (input.shape.size() != 1 || weights.shape.size() != 2 || bias.shape.size() != 1)
		throw std::runtime_error("Linear : only 1D input, 2D weights and 1D bias supported");

	size_t inFeatures = input.shape[0];
	size_t outFeatures = weights.shape[0];

	if (weights.shape[1] != inFeatures || bias.shape[0] != outFeatures)
		throw std::runtime_error("Linear : shape mismatch");

	inp.shape = { inFeatures, 1 };

	GEMM(weights, inp, out);

	out.shape = {outFeatures};

	Add(out, bias, out);
}
