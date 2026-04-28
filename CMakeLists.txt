cmake_minimum_required(VERSION 3.10)

project(cnnRuntime VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_SOURCES SourceFiles/Tensor.cpp SourceFiles/Ops.cpp SourceFiles/Layers.cpp)

set(COMMON_HEADERS Header/Ops.h Header/Layers.h)

add_executable(TestTensor Tests/TestTensor.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
add_executable(TensorOpsBench Benchmarks/TensorOpsBench.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
add_executable(Training Training/Training.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
add_executable(Inference Training/Inference.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
add_executable(Testing Training/Testing.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})

foreach(target TestTensor TensorOpsBench Training Inference Testing)
    target_include_directories(${target} PRIVATE Header/)
endforeach()

set(WEIGHTS_FILE "${CMAKE_SOURCE_DIR}/Training/SavedTensors.bin")
set(TRAIN_DATA "${CMAKE_SOURCE_DIR}/Training/dataset.txt")
set(TEST_DATA "${CMAKE_SOURCE_DIR}/Training/TestDataset.txt")

foreach(target Training Inference Testing)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WEIGHTS_FILE}" $<TARGET_FILE_DIR:${target}>/SavedTensors.bin)
endforeach()

add_custom_command(TARGET Training POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRAIN_DATA}" $<TARGET_FILE_DIR:Training>/dataset.txt)

add_custom_command(TARGET Testing POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEST_DATA}" $<TARGET_FILE_DIR:Testing>/TestDataset.txt)