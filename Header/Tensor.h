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

	void zeros();

	void fillRandom();

	~Tensor();
};

std::ostream& operator << (std::ostream& os, const Tensor& T);