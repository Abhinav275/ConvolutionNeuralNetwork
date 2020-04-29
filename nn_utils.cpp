#include "nn_utils.h"

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

// function to get softmax values
Eigen::VectorXd get_softmax(Eigen::VectorXd y){
	double vectorSum = 0.0;
	for(int i=0;i<y.size();i++){
		y[i] = exp(y[i]);
		vectorSum += y[i];
	}
	y = y/vectorSum;
	return y;  
}

// function to get Euclidean loss
void get_cross_entropy_loss(Eigen::VectorXd y_predict, Eigen::VectorXd y,
	double& loss, Eigen::VectorXd& dl_dy){

	Eigen::VectorXd y_softmax = get_softmax(y_predict);
	for(int i=0;i<y_softmax.size();i++){
		loss += -log(y_softmax(i))*y(i);
	}
	dl_dy = y_softmax - y;
}
