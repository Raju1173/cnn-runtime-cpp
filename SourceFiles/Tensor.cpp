#include "Ops.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <random>

Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape), pData(nullptr), numel(1)
{
	for (size_t dim : shape)
	{
		numel *= dim;
	}

	pData = (float*)std::malloc(numel * sizeof(float));
}

Tensor::Tensor(const Tensor& other) : shape(other.shape), numel(other.numel), pData(nullptr)
{
	pData = (float*)std::malloc(numel * sizeof(float));
	std::copy(other.pData, other.pData + numel, pData);
}

Tensor::Tensor(Tensor&& other) noexcept : shape(std::move(other.shape)), numel(other.numel), pData(other.pData)
{
	other.pData = nullptr;
	other.numel = 0;
}

Tensor& Tensor::operator = (const Tensor& other)
{
	if (this != &other)
	{
		free(pData);
		shape = other.shape;
		numel = other.numel;
		pData = (float*)std::malloc(numel * sizeof(float));
		std::copy(other.pData, other.pData + numel, pData);
	}

	return *this;
}

Tensor::~Tensor()
{
	free(pData);
}

void Tensor::zeros()
{
	for (size_t i = 0; i < numel; i++)
	{
		pData[i] = 0;
	}
}

void Tensor::fillRandom(size_t fanIn)
{
	static std::mt19937 rng(42);
	std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / static_cast<float>(fanIn)));
	for (size_t i = 0; i < numel; i++)
		pData[i] = dist(rng);
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

void SaveTensors(const std::vector<Tensor*>& tensors, const std::string& filename)
{
	FILE* file = fopen(filename.c_str(), "wb");

	if (!file)
		throw std::runtime_error("Failed to open file for writing");

	size_t numTensors = tensors.size();
	fwrite(&numTensors, sizeof(size_t), 1, file);

	for (const Tensor* tensor : tensors)
	{
		size_t numDims = tensor->shape.size();

		fwrite(&numDims, sizeof(size_t), 1, file);
		fwrite(tensor->shape.data(), sizeof(size_t), numDims, file);
		fwrite(tensor->pData, sizeof(float), tensor->numel, file);
	}

	fclose(file);
}

void LoadTensors(std::vector<Tensor*>& tensors, const std::string& filename)
{
	FILE* file = fopen(filename.c_str(), "rb");

	if (!file)
		throw std::runtime_error("Failed to open file for reading");

	size_t numTensors;
	fread(&numTensors, sizeof(size_t), 1, file);

	if (numTensors != tensors.size())
		throw std::runtime_error("Tensor count mismatch");

	for (size_t i = 0; i < numTensors; i++)
	{
		size_t numDims;
		fread(&numDims, sizeof(size_t), 1, file);

		std::vector<size_t> shape(numDims);
		fread(shape.data(), sizeof(size_t), numDims, file);

		Tensor* tensor = tensors[i];

		if (tensor->shape != shape)
			throw std::runtime_error("Shape mismatch during load");

		fread(tensor->pData, sizeof(float), tensor->numel, file);
	}

	fclose(file);
}