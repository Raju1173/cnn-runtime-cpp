#include <cstdlib>
#include <iostream>
#include "Tensor.h"
#include "ExecutionGraph.h"

/*void Execute(Node* Node)
{
	switch (Node->Operation)
	{
		case OpType::ADD :
			add(*(Node->a), *(Node->b), *(Node->out));
			break;
		case OpType::MATMUL:
			GEMM(*(Node->a), *(Node->b), *(Node->out));
			break;
		default:
			throw std::runtime_error("Execute : invalid operation");
			break;
	}
}*/