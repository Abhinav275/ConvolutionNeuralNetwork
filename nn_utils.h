#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <math.h>
using namespace std;

// function to normaliza the images by dividing them by 255
void normalize_eigen(vector<Eigen::VectorXd>& x);

// Utility function to get mini batches
void get_mini_batch(vector<Eigen::VectorXd> x, vector<Eigen::VectorXd> y, int batch_size,
	vector<vector<Eigen::VectorXd>>& mini_x, vector<vector<Eigen::VectorXd>>& mini_y);

// function that implements forward propogation of fully connected layer
Eigen::VectorXd fc(Eigen::VectorXd x, Eigen::MatrixXd w, Eigen::MatrixXd b);

// function to implement back propogation of full connected layer
void fc_backward(Eigen::VectorXd dl_dy, Eigen::VectorXd x, Eigen::MatrixXd w, Eigen::MatrixXd b, Eigen::VectorXd y,
	Eigen::MatrixXd& dl_dx, Eigen::MatrixXd& dl_dw, Eigen::MatrixXd& dl_db);


// function to get Euclidean loss
void get_euclidean_loss(Eigen::VectorXd y_predict, Eigen::VectorXd y,
	double& loss, Eigen::VectorXd& dl_dy);

// function to get softmax values
Eigen::VectorXd get_softmax(Eigen::VectorXd y);

// function to get Euclidean loss
void get_cross_entropy_loss(Eigen::VectorXd y_predict, Eigen::VectorXd y,
	double& loss, Eigen::VectorXd& dl_dy);

// function to get relu of given matrix
Eigen::VectorXd relu(Eigen::VectorXd x);

// function to get relu backward
Eigen::VectorXd relu_backwards(Eigen::VectorXd dl_dy, Eigen::VectorXd x);