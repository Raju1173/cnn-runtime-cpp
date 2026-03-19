#include "Tensor.h"
#include <chrono>
#include <iostream>

void benchmark(size_t N)
{
    Tensor A({ N,N });
    Tensor B({ N,N });
    Tensor C({ N,N });

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
    ms /= runs;

    std::cout << "N=" << N << " avg time: " << ms << " ms\n";
}

int main()
{
    benchmark(64);
    benchmark(128);
    benchmark(256);
    benchmark(512);
}