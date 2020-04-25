////////////////////////////////////////////////////////
// C++ Program to run a convolutional neural network  //
// Author: Abhinav Mehta                              // 
// Email: mehta275@umn.edu                            //
////////////////////////////////////////////////////////

#include <iostream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

// Utility function to tokenize a line for given delimiter
vector<double> getTokens(string line, char delimiter){
	stringstream lineStream(line);
	vector<double> tokens;
	string word;
	
	// traverse the line stream using delimiter
	while(getline(lineStream, word, delimiter)){
		if(delimiter!=' ' || word!="") tokens.push_back((double)stoi(word));
	}

	// return tokens vector
	return tokens;
}

// Utility function to read csv file
void read_data(string filename, vector<vector<double>>& data, vector<int>& labels){
	ifstream inputFile(filename);
	if(inputFile.is_open()){
		string line;
		while(getline(inputFile, line)){
			vector<double> tokens = getTokens(line, ',');
			labels.push_back((int)tokens.back());
			tokens.pop_back();
			data.push_back(tokens);
		}
	}
}

// Utility function to convert data to Eigen Vectors
vector<Eigen::VectorXd> convert_data(vector<vector<double>> vector_data){
	vector<Eigen::VectorXd> eigen_data;
	for(int i=0;i<vector_data.size();i++){
		Eigen::VectorXd temp = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vector_data[i].data(), vector_data[i].size());
		eigen_data.push_back(temp);
	}
	return eigen_data;
}

int main()
{
	vector<vector<double>> im_train;
	vector<int> im_labels;
	read_data("./data/train.csv", im_train, im_labels);
	vector<Eigen::VectorXd> train_data = convert_data(im_train);

	vector<vector<double>> im_test;
	vector<int> im_test_labels;
	read_data("./data/test.csv", im_test, im_test_labels);
	vector<Eigen::VectorXd> test_data = convert_data(im_test);


	return 0;
}