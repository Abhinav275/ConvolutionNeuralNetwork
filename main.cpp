////////////////////////////////////////////////////////
// C++ Program to run a convolutional neural network  //
// Author: Abhinav Mehta                              // 
// Email: mehta275@umn.edu                            //
////////////////////////////////////////////////////////

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

vector<int> getTokens(string line, char delimiter){
	stringstream lineStream(line);
	vector<int> tokens;
	string word;
	
	// traverse the line stream using delimiter
	while(getline(lineStream, word, delimiter)){
		if(delimiter!=' ' || word!="") tokens.push_back(stoi(word));
	}

	// return tokens vector
	return tokens;
}

vector<vector<int>> read_data(string filename){
	vector<vector<int>> result;
	ifstream inputFile(filename);
	if(inputFile.is_open()){
		string line;
		while(getline(inputFile, line)){
			vector<int> tokens = getTokens(line, ',');
			result.push_back(tokens);
		}
	}
	return result;
}

int main()
{
	vector<vector<int>> train_data = read_data("./data/train.csv");
	vector<vector<int>> test_data = read_data("./data/test.csv");
	return 0;
}