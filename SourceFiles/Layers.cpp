#include "Layers.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

void ReLU::forward(Tensor& inp)
{
	this->input = inp;

	for (size_t i = 0; i < inp.numel; i++)
		inp.pData[i] = std::max(inp.pData[i], 0.0f);
}

void ReLU::backward(Tensor& outGrad, Tensor& inGrad)
{
	for (size_t i = 0; i < input.numel; i++)
		inGrad.pData[i] = (input.pData[i] > 0.0f) ? outGrad.pData[i] : 0.0f;
}

Conv2D::Conv2D(size_t inChannels, size_t outChannels, size_t kernelSize) : weights({ outChannels, inChannels, kernelSize, kernelSize }), bias({ outChannels })
{
	weights.fillRandom(inChannels * kernelSize * kernelSize);
	bias.zeros();
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

	this->col = Tensor({ C * R * S, H_out * W_out }); // This should be prellocated somehow...

	Im2col(input, R, S, this->col);

	weights.shape = { K, C * R * S };
	out.shape = { K, H_out * W_out };

	GEMM(weights, col, out);

	for (size_t k = 0; k < K; k++)
	{
		float  B = bias.pData[k];
		float* row = out.pData + k * (H_out * W_out);
		for (size_t i = 0; i < H_out * W_out; i++)
			row[i] += B;
	}

	weights.shape = { K, C, R, S };
	out.shape = { K, H_out, W_out };
}

void Conv2D::backward(const Tensor& outGrad, Tensor& inGrad, Tensor& gradWeights, Tensor& gradBiases)
{
	size_t H = input.shape[1];
	size_t W = input.shape[2];

	size_t K = weights.shape[0];
	size_t C = weights.shape[1];
	size_t R = weights.shape[2];
	size_t S = weights.shape[3];

	size_t H_out = H - R + 1;
	size_t W_out = W - S + 1;
	size_t N_out = H_out * W_out;

	for (size_t k = 0; k < K; k++)
	{
		float sum = 0.0f;
		const float* row = outGrad.pData + k * N_out;
		for (size_t i = 0; i < N_out; i++)
			sum += row[i];
		gradBiases.pData[k] = sum;
	}

	gradWeights.shape = { K, C * R * S };

	for (size_t k = 0; k < K; k++)
	{
		const float* go_row = outGrad.pData + k * N_out;
		for (size_t p = 0; p < C * R * S; p++)
		{
			float sum = 0.0f;
			const float* col_row = col.pData + p * N_out;
			for (size_t n = 0; n < N_out; n++)
				sum += go_row[n] * col_row[n];
			gradWeights.pData[k * C * R * S + p] = sum;
		}
	}

	gradWeights.shape = { K, C, R, S };

	Tensor grad_col({ C * R * S, N_out }); // This should be prellocated somehow...

	weights.shape = { K, C * R * S };

	for (size_t p = 0; p < C * R * S; p++)
	{
		for (size_t n = 0; n < N_out; n++)
		{
			float sum = 0.0f;
			for (size_t k = 0; k < K; k++)
				sum += weights.pData[k * C * R * S + p] * outGrad.pData[k * N_out + n];
			grad_col.pData[p * N_out + n] = sum;
		}
	}

	weights.shape = { K, C, R, S };

	Col2im(grad_col, C, H, W, R, S, inGrad);
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

				float  maxVal = input.pData[c * H * W + in_row * W + in_col];
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

void MaxPool::backward(const Tensor& outGrad, Tensor& inGrad)
{
	for (size_t i = 0; i < inGrad.numel; i++)
		inGrad.pData[i] = 0.0f;

	for (size_t i = 0; i < indices.size(); i++)
		inGrad.pData[indices[i]] += outGrad.pData[i];
}

Linear::Linear(size_t inFeatures, size_t outFeatures) : weights({ outFeatures, inFeatures }), bias({ outFeatures })
{
	weights.fillRandom(inFeatures);
	bias.zeros();
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

	inp.shape = { inFeatures,  1 };
	out.shape = { outFeatures, 1 };

	GEMM(weights, inp, out);

	inp.shape = { inFeatures };
	out.shape = { outFeatures };

	Add(out, bias, out);
}

void Linear::backward(const Tensor& outGrad, Tensor& inGrad, Tensor& gradWeights, Tensor& gradBiases)
{
	size_t inFeatures = weights.shape[1];
	size_t outFeatures = weights.shape[0];

	for (size_t j = 0; j < inFeatures; j++)
	{
		float sum = 0.0f;
		for (size_t i = 0; i < outFeatures; i++)
			sum += weights.pData[i * inFeatures + j] * outGrad.pData[i];
		inGrad.pData[j] = sum;
	}

	for (size_t i = 0; i < outFeatures; i++)
		for (size_t j = 0; j < inFeatures; j++)
			gradWeights.pData[i * inFeatures + j] = outGrad.pData[i] * input.pData[j];

	for (size_t i = 0; i < outFeatures; i++)
		gradBiases.pData[i] = outGrad.pData[i];
}

void Softmax::forward(const Tensor& inp, Tensor& out)
{
	if (inp.shape.size() != 1)
		throw std::runtime_error("Softmax: only 1-D input supported");

	size_t N = inp.numel;

	float maxVal = *std::max_element(inp.pData, inp.pData + N);

	float sumExp = 0.0f;
	for (size_t i = 0; i < N; i++)
	{
		out.pData[i] = std::exp(inp.pData[i] - maxVal);
		sumExp += out.pData[i];
	}

	for (size_t i = 0; i < N; i++)
		out.pData[i] /= sumExp;

	this->probs = out;
}

void Softmax::backward(const Tensor& outGrad, Tensor& inGrad)
{
	size_t N = probs.numel;

	float dot = 0.0f;
	for (size_t i = 0; i < N; i++)
		dot += outGrad.pData[i] * probs.pData[i];

	for (size_t i = 0; i < N; i++)
		inGrad.pData[i] = probs.pData[i] * (outGrad.pData[i] - dot);
}

float CrossEntropyLoss::forward(const Tensor& probs, size_t label)
{
	if (label >= probs.numel)
		throw std::runtime_error("CrossEntropyLoss: label out of range");

	float p = std::max(probs.pData[label], 1e-7f);
	return -std::log(p);
}

float SoftmaxCrossEntropyLoss::forward(const Tensor& logits, size_t label)
{
	if (logits.shape.size() != 1)
		throw std::runtime_error("SoftmaxCrossEntropyLoss: only 1-D logits supported");
	if (label >= logits.numel)
		throw std::runtime_error("SoftmaxCrossEntropyLoss: label out of range");

	size_t N = logits.numel;

	float maxVal = *std::max_element(logits.pData, logits.pData + N);

	probs = Tensor({ N });

	float sumExp = 0.0f;
	for (size_t i = 0; i < N; i++)
	{
		probs.pData[i] = std::exp(logits.pData[i] - maxVal);
		sumExp += probs.pData[i];
	}

	for (size_t i = 0; i < N; i++)
		probs.pData[i] /= sumExp;

	float p = std::max(probs.pData[label], 1e-7f);
	return -std::log(p);
}

void SoftmaxCrossEntropyLoss::backward(Tensor& inGrad, size_t label)
{
	if (label >= probs.numel)
		throw std::runtime_error("SoftmaxCrossEntropyLoss: label out of range");

	for (size_t i = 0; i < probs.numel; i++)
		inGrad.pData[i] = probs.pData[i];

	inGrad.pData[label] -= 1.0f;
}