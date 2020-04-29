#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <vector>
using namespace std;

// Utility function to tokenize a line for given delimiter
vector<double> getTokens(string line, char delimiter);

// Utility function to read csv file
void read_data(string filename, vector<vector<double>>& data, vector<int>& labels);

// Utility function to convert data to Eigen Vectors
vector<Eigen::VectorXd> convert_data(vector<vector<double>> vector_data);

// Utility function to convert labels to one hot encoding
vector<Eigen::VectorXd> convert_labels(vector<int> labels, int size);

// function to generate a random matrix with given range and size
Eigen::MatrixXd get_random_matrix(double low, double high, int m, int n);