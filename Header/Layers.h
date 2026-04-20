#pragma once
#include "Ops.h"

class ReLU
{
    Tensor input;

public:
    void forward(Tensor& inp);
    void backward(Tensor& outGrad, Tensor& inGrad);
};

class Conv2D
{
public:
    Tensor weights;
    Tensor bias;

    Tensor input;
    Tensor col;

    Conv2D(size_t inChannels, size_t outChannels, size_t kernelSize);

    void forward(const Tensor& inp, Tensor& out);
    void backward(const Tensor& outGrad, Tensor& inGrad, Tensor& gradWeights, Tensor& gradBiases);
};

class MaxPool
{
public:
    std::vector<size_t> indices;

    void forward(const Tensor& input, Tensor& out);
    void backward(const Tensor& outGrad, Tensor& inGrad);
};

class Linear
{
public:
    Tensor weights;
    Tensor bias;

    Tensor input;

    Linear(size_t inFeatures, size_t outFeatures);

    void forward(Tensor& inp, Tensor& out);
    void backward(const Tensor& outGrad, Tensor& inGrad, Tensor& gradWeights, Tensor& gradBiases);
};

class Softmax
{
    Tensor probs;

public:

    void forward(const Tensor& inp, Tensor& out);
    void backward(const Tensor& outGrad, Tensor& inGrad);
};

class CrossEntropyLoss
{
public:
    float forward(const Tensor& probs, size_t label);
};

class SoftmaxCrossEntropyLoss
{
    Tensor probs;

public:

    float forward(const Tensor& logits, size_t label);
    void backward(Tensor& inGrad, size_t label);
};