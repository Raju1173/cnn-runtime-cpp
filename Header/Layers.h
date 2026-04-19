#include "Ops.h"

class ReLU
{
    Tensor input;

public:
    void forward(Tensor& inp);
    void backward(Tensor& gradOutput, Tensor& gradInput);
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
};

class MaxPool
{
public:
    std::vector<size_t> indices;

    void forward(const Tensor& input, Tensor& out);
};

class Linear
{
public:
    Tensor weights;
    Tensor bias;

    Tensor input;

    Linear(size_t inFeatures, size_t outFeatures);

    void forward(Tensor& inp, Tensor& out);
};