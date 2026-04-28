#include "Layers.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

int main(int argc, char** argv)
{
    std::vector<std::string> classNames = {"apple","car","cat","chair","clock","cup","dog","fish","house","tree"};

    Conv2D conv1(1, 16, 3);
    ReLU relu1;
    MaxPool pool1;

    Conv2D conv2(16, 32, 3);
    ReLU relu2;
    MaxPool pool2;

    Conv2D conv3(32, 64, 3);
    ReLU relu3;

    Linear fc1(64 * 3 * 3, 128);
    ReLU relu4;
    Linear fc2(128, classNames.size());

    std::vector<Tensor*> Model = {&conv1.weights, &conv1.bias,&conv2.weights, &conv2.bias,&conv3.weights, &conv3.bias,&fc1.weights, &fc1.bias,&fc2.weights, &fc2.bias};

    LoadTensors(Model, "SavedTensors.bin");

    Tensor input({ 1, 28, 28 });

    FILE* inf = fopen(argv[1], "rb");

    if (!inf)
    {
        fprintf(stderr, "Cannot open input file\n"); 
        
        return 1;
    }

    fread(input.pData, sizeof(float), 784, inf);
    fclose(inf);

    Tensor conv1Out({ 16, 26, 26 });
    Tensor pool1Out({ 16, 13, 13 });

    Tensor conv2Out({ 32, 11, 11 });
    Tensor pool2Out({ 32, 5, 5 });

    Tensor conv3Out({ 64, 3, 3 });

    Tensor fc1Out({ 128 });
    Tensor logits({ classNames.size() });

    conv1.forward(input, conv1Out);
    relu1.forward(conv1Out);
    pool1.forward(conv1Out, pool1Out);

    conv2.forward(pool1Out, conv2Out);
    relu2.forward(conv2Out);
    pool2.forward(conv2Out, pool2Out);

    conv3.forward(pool2Out, conv3Out);
    relu3.forward(conv3Out);

    conv3Out.shape = { 64 * 3 * 3 };
    fc1.forward(conv3Out, fc1Out);
    conv3Out.shape = { 64, 3, 3 };

    relu4.forward(fc1Out);
    fc2.forward(fc1Out, logits);

    size_t pred = 0;

    for (size_t i = 1; i < classNames.size(); i++)
        if (logits.pData[i] > logits.pData[pred]) pred = i;

    printf("%s\n", classNames[pred].c_str());

    return 0;
}