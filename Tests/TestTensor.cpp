#include "Layers.h"
#include <iostream>

void TestGEMM()
{
	Tensor A( {3, 2} );
	Tensor B( {2, 4} );

	A.pData[0] = 1; A.pData[1] = 2;
	A.pData[2] = 3; A.pData[3] = 4;
	A.pData[4] = 5; A.pData[5] = 6;

	B.pData[0] = 1; B.pData[1] = 2; B.pData[2] = 3; B.pData[3] = 4;
	B.pData[4] = 5; B.pData[5] = 6; B.pData[6] = 7; B.pData[7] = 8;

	Tensor C({ 3, 4 });

	GEMM(A, B, C);

	std::cout << "GEMM Output : \n" << C;
}

void TestIm2Col()
{
    Tensor input({1, 4, 4 });

	Tensor col({ 4, 9 });

    for (int i = 0; i < 16; i++)
        input.pData[i] = i + 1;

    Im2col(input, 2, 2, col);

    std::cout << "\nIm2Col Output : \n" << col;
}

void TestConv2D()
{
	Tensor input({ 1, 4, 4 });

	Tensor output({ 1, 2, 2 });

	Tensor weights({ 1, 1, 3, 3 });
	Tensor biases({ 1 });

	weights.pData[0] = 1; weights.pData[1] = 0; weights.pData[2] = 0;
	weights.pData[3] = 0; weights.pData[4] = 1; weights.pData[5] = 0;

	biases.pData[0] = 0;

	float data[] = {
		1,3,2,1,
		4,6,5,2,
		7,8,9,3,
		1,2,3,4
	};

	for (int i = 0; i < 16; i++)
		input.pData[i] = data[i];

	Conv2D conv(1, 1, 3);
	conv.weights = weights;
	conv.bias = biases;

	conv.forward(input, output);

	std::cout << "\nConv2D Output : \n" << output;
}

void TestMaxPool()
{
    Tensor input({ 1, 4, 4 });
	Tensor output({ 1, 2, 2 });

    float data[] = {
        1,3,2,1,
        4,6,5,2,
        7,8,9,3,
        1,2,3,4
    };

    for (int i = 0; i < 16; i++)
        input.pData[i] = data[i];

	MaxPool pool;

    pool.forward(input, output);

    std::cout << "\nMaxPool Output : \n" << output << std::endl;
}

void TestConv2DBackward()
{
	Tensor input({ 1, 4, 4 });
	Tensor output({ 1, 2, 2 });
	Tensor outGrad({ 1, 2, 2 });
	Tensor weights({ 1, 1, 3, 3 });
	Tensor gradWeights({ 1, 1, 3, 3 });
	Tensor biases({ 1 });
	Tensor gradBiases({ 1 });

	weights.pData[0] = 1; weights.pData[1] = 0; weights.pData[2] = 0;
	weights.pData[3] = 0; weights.pData[4] = 1; weights.pData[5] = 0;
	biases.pData[0] = 0;

	float data[] = {
		1,3,2,1,
		4,6,5,2,
		7,8,9,3,
		1,2,3,4
	};

	for (int i = 0; i < 16; i++)
		input.pData[i] = data[i];

	outGrad.pData[0] = 1; outGrad.pData[1] = 2;
	outGrad.pData[2] = 3; outGrad.pData[3] = 4;

	Conv2D conv(1, 1, 3);

	conv.weights = weights;
	conv.bias = biases;

	conv.forward(input, output);
	conv.backward(outGrad, input, gradWeights, gradBiases);

	std::cout << "\nConv2D Backward Weights : \n" << gradWeights;
	std::cout << "\nConv2D Backward Biases : \n" << gradBiases;
}

void TestMaxPoolBackward()
{
	Tensor input({ 1, 4, 4 });
	Tensor output({ 1, 2, 2 });
	Tensor outGrad({ 1, 2, 2 });
	Tensor inGrad({ 1, 4, 4 });

	float data[] = {
		1,3,2,1,
		4,6,5,2,
		7,8,9,3,
		1,2,3,4
	};

	for (int i = 0; i < 16; i++)
		input.pData[i] = data[i];

	outGrad.pData[0] = 1; outGrad.pData[1] = 2;
	outGrad.pData[2] = 3; outGrad.pData[3] = 4;

	MaxPool pool;

	pool.forward(input, output);
	pool.backward(outGrad, inGrad);

	std::cout << "\nMaxPool Backward : \n" << inGrad << std::endl;
}

void TestSCELBackward()
{
	SoftmaxCrossEntropyLoss SCEL;

	Tensor logits({ 10 });
	Tensor outGrad({ 10 });

	for (int i = 0; i < 10; i++)
		logits.pData[i] = static_cast<float>(i);

	size_t label = 3;

	float loss = SCEL.forward(logits, label);
	SCEL.backward(outGrad, label);

	std::cout << "\nSoftmaxCrossEntropyLoss Backward : \n" << outGrad << std::endl;
}

int main()
{
	TestGEMM();
    TestIm2Col();
    TestConv2D();
	TestConv2DBackward();
    TestMaxPool();
    TestMaxPoolBackward();
	TestSCELBackward();
}