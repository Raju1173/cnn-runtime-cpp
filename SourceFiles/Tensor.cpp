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
