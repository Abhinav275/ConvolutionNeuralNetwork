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

// function to normaliza the images by dividing them by 255
void normalize_eigen(vector<Eigen::VectorXd>& x){
	for(int i=0;i<x.size();i++) x[i] = x[i]/255.0;
	return; 
}

// Utility function to get mini batches
void get_mini_batch(vector<Eigen::VectorXd> x, vector<Eigen::VectorXd> y, int batch_size,
	vector<vector<Eigen::VectorXd>>& mini_x, vector<vector<Eigen::VectorXd>>& mini_y){
	
	// initialize data and shuffle
	vector<pair<Eigen::VectorXd, Eigen::VectorXd>> data;
	for(int i=0;i<x.size();i++)
		data.push_back({x[i], y[i]});
	random_shuffle(data.begin(), data.end());
	
	// get number of batches
	int size = x.size()/batch_size;

	// add batches
	int k=0;
	for(int i=0;i<size;i++){
		mini_x.push_back({});
		mini_y.push_back({});
		for(int j=0;j<batch_size;j++){
			mini_x[i].push_back(data[k].first);
			mini_y[i].push_back(data[k].second);
			k++;
			if(k==data.size())
				break;
		}
		if(k==data.size())
			break;
	}
	return;
}

// function to generate a random matrix with given range and size
Eigen::MatrixXd get_random_matrix(double low, double high, int m, int n){
	double range = high - low;
	Eigen::MatrixXd mat = Eigen::MatrixXd::Random(m,n);
	mat = (mat + Eigen::MatrixXd::Constant(m, n, 1.0))*range/2.0;
	mat = (mat + Eigen::MatrixXd::Constant(m, n, low));
	return mat;
}

// function that implements forward propogation of fully connected layer
Eigen::VectorXd fc(Eigen::VectorXd x, Eigen::MatrixXd w, Eigen::MatrixXd b){
	Eigen::VectorXd y = w*x+b;
	return y;
}

// function to implement back propogation of full connected layer
void fc_backward(Eigen::VectorXd dl_dy, Eigen::VectorXd x, Eigen::MatrixXd w, Eigen::MatrixXd b, Eigen::VectorXd y,
	Eigen::MatrixXd& dl_dx, Eigen::MatrixXd& dl_dw, Eigen::MatrixXd& dl_db){
	dl_dx = w.transpose()*dl_dy;
	dl_dw = dl_dy*x.transpose();
	dl_db = dl_dy;
	return;
}

// function to get Euclidean loss
void get_euclidean_loss(Eigen::VectorXd y_predict, Eigen::VectorXd y,
	double& loss, Eigen::VectorXd& dl_dy){
	dl_dy = y_predict - y;
	loss = dl_dy.norm();
	return;
}

// function to run single layer perceptron with no activation
void train_slp_linear(vector<vector<Eigen::VectorXd>>& mini_x, vector<vector<Eigen::VectorXd>>& mini_y,
	Eigen::MatrixXd& w, Eigen::MatrixXd& b){
	double lr = 0.01;
	double decay = 0.9;

	// initialize weights
	w = get_random_matrix(0.0, 0.1, 10, mini_x[0][0].size());
	b = get_random_matrix(0.0, 0.1, 10, 1);

	// batch number and batch, epoch loss;
	int k=0;
	double batch_loss = 0.0;
	double epoch_loss = 0.0;

	cout<<"Training Single Layer Perceptron"<<endl;
	for(int iter = 1; iter<10000; iter++){
		if(iter%2000 == 0) lr = lr*decay;
		Eigen::MatrixXd dL_dw = Eigen::MatrixXd::Zero(w.rows(), w.cols());
		Eigen::MatrixXd dL_db = Eigen::MatrixXd::Zero(b.rows(), b.cols());

		// iterate over current batch;
		for(int i=0;i<mini_x[k].size();i++){
			Eigen::VectorXd y_tilde = fc(mini_x[k][i], w, b);
			double loss;
			Eigen::VectorXd dl_dy;
			get_euclidean_loss(y_tilde, mini_y[k][i], loss, dl_dy);

			batch_loss += loss;

			// initialize gradients
			Eigen::MatrixXd dl_dx = Eigen::MatrixXd::Zero(mini_x[k][i].rows(), mini_x[k][i].cols());
			Eigen::MatrixXd dl_dw = Eigen::MatrixXd::Zero(w.rows(), w.cols());
			Eigen::MatrixXd dl_db = Eigen::MatrixXd::Zero(b.rows(), b.cols());


			// backpropogate gradients
			fc_backward(dl_dy, mini_x[k][i], w, b, mini_y[k][i],
				dl_dx, dl_dw, dl_db);

			// add the gradients for this batch
			dL_dw = dL_dw + dl_dw;
			dL_db = dL_db + dl_db;
		}

		// update weights
		w = w - dL_dw*(lr/mini_x[k].size());
		b = b - dL_db*(lr/mini_x[k].size());

		// cout<<batch_loss/mini_x[k].size()<<endl;
		epoch_loss += batch_loss/mini_x[k].size();
		batch_loss = 0.0;
		k+=1;
		if(k == mini_x.size()) {
			cout<<"Epoch loss: "<<epoch_loss/mini_x.size()<<endl;
			k=0;
			epoch_loss = 0.0;
		}

	}

	return;
}

// function to test single layer perceptron
void test_slp_linear(Eigen::MatrixXd w, Eigen::MatrixXd b, vector<Eigen::VectorXd> test_data, vector<int> test_labels){
	double accuracy = 0.0;
	cout<<"Testing Single Layer Perceptron"<<endl;
	for(int i=0;i<test_data.size();i++){
		
		// forward propogation in network
		Eigen::VectorXd y_predict = fc(test_data[i], w, b);
		
		// get index of max value
		int y_predict_label;
		y_predict.maxCoeff(&y_predict_label);

		if(y_predict_label == test_labels[i]) accuracy+=1.0;
	}

	cout<<"Model Accuracy: "<<(accuracy/test_data.size())*100<<endl;
	return;
}


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

	Eigen::MatrixXd w,b;

	train_slp_linear(mini_x, mini_y,
		w,b);

	test_slp_linear(w, b, test_data, im_test_labels);

	return 0;
}