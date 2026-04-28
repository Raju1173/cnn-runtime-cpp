#include "Layers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>

int main()
{
    const char* weightsPath = "SavedTensors.bin";
    const char* dataPath = "TestDataset.txt";

    const std::vector<std::string> classNames = {"apple","car","cat","chair","clock","cup","dog","fish","house","tree"};

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

    std::vector<Tensor*> Model = {&conv1.weights, &conv1.bias,&conv2.weights, &conv2.bias,&conv3.weights, &conv3.bias,&fc1.weights, &fc1.bias,&fc2.weights, &fc2.bias};

    LoadTensors(Model, weightsPath);

    std::vector<std::vector<float>> images;
    std::vector<size_t> labels;

    std::ifstream file(dataPath);
    if (!file.is_open()) { fprintf(stderr, "Cannot open dataset\n"); return 1; }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        for (char& c : line) if (c == ',') c = ' ';

        std::istringstream iss(line);
        size_t lbl;
        if (!(iss >> lbl)) continue;

        labels.push_back(lbl);
        std::vector<float> img(784);

        for (size_t i = 0; i < 784; i++) iss >> img[i];
        images.push_back(img);
    }

    std::cout << "Dataset loaded: " << images.size() << "\n";

    Tensor input({ 1, 28, 28 });

    Tensor conv1Out({ 16, 26, 26 });
    Tensor pool1Out({ 16, 13, 13 });

    Tensor conv2Out({ 32, 11, 11 });
    Tensor pool2Out({ 32, 5, 5 });

    Tensor conv3Out({ 64, 3, 3 });

    Tensor fc1Out({ 128 });
    Tensor logits({ 10 });

    size_t correct = 0;

    std::vector<size_t> classTotal(10, 0);
    std::vector<size_t> classCorrect(10, 0);

    for (size_t i = 0; i < images.size(); i++)
    {
        std::copy(images[i].begin(), images[i].end(), input.pData);
        size_t label = labels[i];

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
        for (size_t c = 1; c < 10; c++)
            if (logits.pData[c] > logits.pData[pred]) pred = c;

        classTotal[label]++;
        if (pred == label)
        {
            correct++;
            classCorrect[label]++;
        }
    }

    float acc = 100.0f * correct / images.size();
    std::cout << "Accuracy: " << acc << "%\n";

    return 0;
}