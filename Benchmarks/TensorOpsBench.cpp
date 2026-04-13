#include "Tensor.h"
#include <chrono>
#include <iostream>

void BenchGEMM(size_t N)
{
    Tensor A({ N,N });
    Tensor B({ N,N });

    A.fillRandom();
    B.fillRandom();

    const int runs = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0;i < runs;i++)
    {
        GEMM(A, B);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "N=" << N << " avg time: " << ms / runs << " ms\n";
}

void BenchIm2Col(size_t N)
{
	Tensor input({ 3,N,N });
	Tensor weights({ 64,3,3,3 });
	Tensor bias({ 64 });

	input.fillRandom();
	weights.fillRandom();
	bias.fillRandom();

	const int runs = 1000;

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0;i < runs;i++)
	{
		Im2col(input, 3, 3);
	}

	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "N=" << N << " avg time: " << ms / runs << " ms\n";
}

int main()
{
	std::cout << "Benchmarking GEMM:\n";
    BenchGEMM(64);
    BenchGEMM(128);
    BenchGEMM(256);
    BenchGEMM(512);

    std::cout << "\nBenchmarking Im2Col:\n";
    BenchIm2Col(64);
    BenchIm2Col(128);
    BenchIm2Col(256);
    BenchIm2Col(512);
}