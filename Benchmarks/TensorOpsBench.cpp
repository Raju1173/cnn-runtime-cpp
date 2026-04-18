#include "Layers.h"
#include <chrono>
#include <iostream>

void BenchGEMM(size_t N)
{
    Tensor A({ N, N });
    Tensor B({ N, N });
    Tensor C({ N, N });

    A.fillRandom();
    B.fillRandom();

    const int runs = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0;i < runs;i++)
    {
        GEMM(A, B, C);
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

	Tensor output({ 64, (N - 2) * (N - 2) });

	input.fillRandom();
	weights.fillRandom();
	bias.fillRandom();

	const int runs = 1000;

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0;i < runs;i++)
	{
		Im2col(input, 3, 3, output);
	}

	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "N=" << N << " avg time: " << ms / runs << " ms\n";
}

void BenchConv2D(size_t N)
{
	Tensor input({ 3,N,N });
	Tensor output({ 64, N - 2, N - 2 });

	input.fillRandom();

	Conv2D conv(3, 64, 3);

	const int runs = 1000;

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0;i < runs;i++)
	{
		conv.forward(input, output);
	}

	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "N=" << N << " avg time: " << ms / runs << " ms\n";
}

void BenchMaxPool(size_t N)
{
	Tensor input({ 3,N,N });
	Tensor output({ 3, N/2, N/2 });

	const int runs = 1000;

	auto start = std::chrono::high_resolution_clock::now();

	MaxPool pool;

	for (int i = 0;i < runs;i++)
	{
		pool.forward(input, output);
	}

	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "N=" << N << " avg time: " << ms / runs << " ms\n";
}

void RunBench(const std::string& name, void(*bench)(size_t))
{
	std::cout << "\nBenchmarking " << name << ":\n";

	for (int size : {64, 128, 256, 512})
		bench(size);
}

void Benchmark(bool gemm = true, bool im2Col = true, bool conv2D = true, bool maxPool = true)
{
	if (gemm)   RunBench("GEMM", BenchGEMM);
	if (im2Col) RunBench("Im2Col", BenchIm2Col);
	if (conv2D) RunBench("Conv2D", BenchConv2D);
	if (maxPool)RunBench("MaxPool", BenchMaxPool);
}

int main()
{
	Benchmark(1, 1, 1, 1);
}
