#pragma once
#include<vector>
#include<cstddef>
#include<iostream>

class Tensor
{
public :
	float *pData;
	std::vector<size_t> shape;
	size_t numel;

	Tensor() { pData = nullptr; numel = 0; };
	Tensor(const std::vector<size_t>& shape);

	Tensor(const Tensor& other);
	Tensor(Tensor&& other) noexcept;
	Tensor& operator = (const Tensor& other);

	void zeros();

	void fillRandom(size_t fanIn);

	~Tensor();
};

std::ostream& operator << (std::ostream& os, const Tensor& T);