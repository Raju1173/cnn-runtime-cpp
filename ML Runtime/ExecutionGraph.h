#pragma once
#include <vector>
#include "Tensor.h"

enum OpType
{
	ADD,
	MATMUL
};

struct Node
{
	OpType Operation;
	Tensor* a;
	Tensor* b;
	Tensor* out;

	std::vector<Node> Nodes;
};

void Execute(Node* Node);