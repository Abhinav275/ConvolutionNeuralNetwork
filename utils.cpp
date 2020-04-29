#include "utils.h"

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

// Utility function to convert labels to one hot encoding
vector<Eigen::VectorXd> convert_labels(vector<int> labels, int size){
	vector<Eigen::VectorXd> new_labels;
	for(int i=0;i<labels.size();i++){
		Eigen::VectorXd temp = Eigen::VectorXd::Zero(size);
		temp(labels[i]) = 1.0;
		new_labels.push_back(temp);
	}
	return new_labels;
}

// function to generate a random matrix with given range and size
Eigen::MatrixXd get_random_matrix(double low, double high, int m, int n){
	double range = high - low;
	Eigen::MatrixXd mat = Eigen::MatrixXd::Random(m,n);
	mat = (mat + Eigen::MatrixXd::Constant(m, n, 1.0))*range/2.0;
	mat = (mat + Eigen::MatrixXd::Constant(m, n, low));
	return mat;
}
