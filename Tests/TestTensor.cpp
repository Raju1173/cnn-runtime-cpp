#include "Tensor.h"
#include <iostream>

void testGEMM()
{
	Tensor A( {3, 2} );
	Tensor B( {2, 4} );

	A.pData[0] = 1; A.pData[1] = 2;
	A.pData[2] = 3; A.pData[3] = 4;
	A.pData[4] = 5; A.pData[5] = 6;

	B.pData[0] = 1; B.pData[1] = 2; B.pData[2] = 3; B.pData[3] = 4;
	B.pData[4] = 5; B.pData[5] = 6; B.pData[6] = 7; B.pData[7] = 8;

	GEMM(A, B).print();
}

int main()
{
	testGEMM();
}