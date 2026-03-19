#include <iostream>
#include "Tensor.h"
#include "ExecutionGraph.h"

/*int main()
{
	Tensor a({ 2, 3 });
	Tensor b({ 3, 6 });
	Tensor c({ 2, 6 });

	Node n{ OpType::MATMUL, &a, &b, &c };

	Execute(&n);

	for (int i = 0; i < c.numel; i++)
	{
		std::cout << c.pData[i];
	}

	return 0;
}*/
