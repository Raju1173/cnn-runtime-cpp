cmake_minimum_required(VERSION 3.10)

project(cnnRuntime VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TestTensor Tests/TestTensor.cpp SourceFiles/Tensor.cpp)
add_executable(TensorOpsBench Benchmarks/TensorOpsBench.cpp SourceFiles/Tensor.cpp)

target_include_directories(TestTensor PRIVATE Header/)
target_include_directories(TensorOpsBench PRIVATE Header/)