#include "Layers.h"
#include <numeric>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

int main()
{
    std::ifstream file("dataset.txt");

    std::vector<std::vector<float>> images;
    std::vector<size_t> trainLabels;

    std::string line;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);

        size_t label;
        iss >> label;
        trainLabels.push_back(label);

        std::vector<float> img(28 * 28);

        for (size_t i = 0; i < 28 * 28; i++)
            iss >> img[i];

        images.push_back(img);
    }

    const size_t numEpochs = 5;
    const size_t numSamples = images.size();
    const float  lr = 0.001f;

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
    Linear fc2(128, 10);

    SoftmaxCrossEntropyLoss SCEL;

    Tensor input({ 1, 28, 28 });

    Tensor conv1Out({ 16, 26, 26 });
    Tensor pool1Out({ 16, 13, 13 });

    Tensor conv2Out({ 32, 11, 11 });
    Tensor pool2Out({ 32, 5, 5 });

    Tensor conv3Out({ 64, 3, 3 });

    Tensor fc1Out({ 128 });
    Tensor logits({ 10 });

    Tensor gConv1W({ 16, 1, 3, 3 });
    Tensor gConv1B({ 16 });

    Tensor gConv2W({ 32, 16, 3, 3 });
    Tensor gConv2B({ 32 });

    Tensor gConv3W({ 64, 32, 3, 3 });
    Tensor gConv3B({ 64 });

    Tensor gFc1W({ 128, 64 * 3 * 3 });
    Tensor gFc1B({ 128 });

    Tensor gFc2W({ 10, 128 });
    Tensor gFc2B({ 10 });

    Tensor gLogits({ 10 });
    Tensor gFc1Out({ 128 });

    Tensor gConv3Out({ 64, 3, 3 });
    Tensor gPool2Out({ 32, 5, 5 });
    Tensor gConv2Out({ 32, 11, 11 });
    Tensor gPool1Out({ 16, 13, 13 });
    Tensor gConv1Out({ 16, 26, 26 });
    Tensor gInput({ 1, 28, 28 });

    std::mt19937 rng(42);

    std::vector<size_t> order(numSamples);
    std::iota(order.begin(), order.end(), 0);

    std::vector<Tensor*> Model = {&conv1.weights, &conv1.bias, &conv2.weights, &conv2.bias, &conv3.weights, &conv3.bias, &fc1.weights, &fc1.bias, &fc2.weights, &fc2.bias};

    //LoadTensors(Model, "SavedTensors.bin");

    for (size_t epoch = 0; epoch < numEpochs; epoch++)
    {
        std::shuffle(order.begin(), order.end(), rng);

        float epochLoss = 0.0f;
        size_t correct = 0;

        for (size_t idx : order)
        {
            std::copy(images[idx].begin(), images[idx].end(), input.pData);
            size_t label = trainLabels[idx];

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

            float loss = SCEL.forward(logits, label);
            epochLoss += loss;

            size_t pred = 0;
            for (size_t i = 1; i < 10; i++)
                if (logits.pData[i] > logits.pData[pred]) pred = i;

            if (pred == label) correct++;

            gConv1W.zeros(); gConv1B.zeros();
            gConv2W.zeros(); gConv2B.zeros();
            gConv3W.zeros(); gConv3B.zeros();
            gFc1W.zeros();  gFc1B.zeros();
            gFc2W.zeros();  gFc2B.zeros();

            SCEL.backward(gLogits, label);

            fc2.backward(gLogits, gFc1Out, gFc2W, gFc2B);

            relu4.backward(gFc1Out, gFc1Out);

            gConv3Out.shape = { 64 * 3 * 3 };
            fc1.backward(gFc1Out, gConv3Out, gFc1W, gFc1B);
            gConv3Out.shape = { 64, 3, 3 };

            relu3.backward(gConv3Out, gConv3Out);

            conv3.backward(gConv3Out, gPool2Out, gConv3W, gConv3B);

            pool2.backward(gPool2Out, gConv2Out);

            relu2.backward(gConv2Out, gConv2Out);

            conv2.backward(gConv2Out, gPool1Out, gConv2W, gConv2B);

            pool1.backward(gPool1Out, gConv1Out);

            relu1.backward(gConv1Out, gConv1Out);

            conv1.backward(gConv1Out, gInput, gConv1W, gConv1B);

            SGD(conv1.weights, gConv1W, lr);
            SGD(conv1.bias, gConv1B, lr);

            SGD(conv2.weights, gConv2W, lr);
            SGD(conv2.bias, gConv2B, lr);

            SGD(conv3.weights, gConv3W, lr);
            SGD(conv3.bias, gConv3B, lr);

            SGD(fc1.weights, gFc1W, lr);
            SGD(fc1.bias, gFc1B, lr);

            SGD(fc2.weights, gFc2W, lr);
            SGD(fc2.bias, gFc2B, lr);
        }

        float avgLoss = epochLoss / (float)numSamples;
        float acc = 100.0f * (float)correct / (float)numSamples;

        std::cout << "Epoch " << (epoch + 1) << " / " << numEpochs << " loss: " << avgLoss << " acc: " << acc << "%\n";
    }

    SaveTensors(Model, "SavedTensors.bin");

    return 0;
}