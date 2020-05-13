////////////////////////////////////////////////////////
// C++ Program to run a convolutional neural network  //
// Author: Abhinav Mehta                              // 
// Email: mehta275@umn.edu                            //
////////////////////////////////////////////////////////

#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <math.h>
#include "utils.h"
#include "nn_utils.h"
#include "SLPLinear.cpp"
#include "SingleLayerPerceptron.cpp"
#include "MultiLayerPerceptron.cpp"
#include "ConvolutionalNeuralNetwork.cpp"

using namespace std;

int main()
{
	vector<vector<double>> im_train;
	vector<int> im_labels;
	read_data("./data/train.csv", im_train, im_labels);
	vector<Eigen::VectorXd> train_data = convert_data(im_train);
	vector<Eigen::VectorXd> train_labels = convert_labels(im_labels, 10);
	normalize_eigen(train_data);

	vector<vector<double>> im_test;
	vector<int> im_test_labels;
	read_data("./data/test.csv", im_test, im_test_labels);
	vector<Eigen::VectorXd> test_data = convert_data(im_test);
	normalize_eigen(test_data);

	int batch_size = 32;

	vector<vector<Eigen::VectorXd>> mini_x;
	vector<vector<Eigen::VectorXd>> mini_y;
	get_mini_batch(train_data, train_labels, batch_size, mini_x, mini_y);

	// SLPLinear slpLinear;
	// slpLinear.train(mini_x, mini_y);
	// slpLinear.test(test_data, im_test_labels);

	// SingleLayerPerceptron singleLayerPerceptron;
	// singleLayerPerceptron.train(mini_x, mini_y);
	// singleLayerPerceptron.test(test_data, im_test_labels);

	// MultiLayerPerceptron multiLayerPerceptron;
	// multiLayerPerceptron.train(mini_x, mini_y);
	// multiLayerPerceptron.test(test_data, im_test_labels);

	ConvolutionalNeuralNetwork cnn;
	cnn.train(mini_x, mini_y);
	cnn.test(test_data, im_test_labels);

	return 0;
}