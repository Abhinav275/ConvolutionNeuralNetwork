# ConvolutionalNeuralNetwork
[![GitHub license](https://img.shields.io/github/license/Abhinav275/ConvolutionalNeuralNetwork)](https://github.com/Abhinav275/ConvolutionalNeuralNetwork/blob/master/LICENSE)

Implementation of convolutional neural network in C++.

# Requirments
```Eigen3``` is required for Tensor and Matrix operations.

# Dataset
I have used small MNIST dataset which has ```12000``` training images and ```2000``` test images of size ```14x14```. The dataset was preprocessed and saved in csv file where ```14x14``` images where flattened to ```196x1``` vector and the label was appended as the last element. So finally each record is the dataset is of size ```197x1```.

In order to train with a different data, one would need to preprocess the data and save it in a csv file as mentioned above. There might be some dimensional changes required. 

# Implementation Details
I have implemented ```Single Layer Linear Neural Network```, ```Single Layer Perceptron```, ```Multi Layer Perceptron``` and ```Convolutional Neural Network``` in their respective classes. Each class has train and test functions. Train functions implement ```mini batch gradient descent``` therefore, it expects mini batches of images as input.

A utility function ```get_mini_batch``` has also been implemented to get batches for given size.

All the neural network and machine learning functions like ```cross entropy loss```, ```convolution```, ```relu```, ```pool2x2```, ```fc``` etc. along with their back propogation have been implemented in ```nn_utils```.
