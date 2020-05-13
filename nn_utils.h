#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <math.h>
#include <unsupported/Eigen/CXX11/Tensor>
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

// function to pad an image with zeros
Eigen::MatrixXd pad(Eigen::MatrixXd img);

// function to convulate over given image
Eigen::Tensor<double, 3> conv(Eigen::MatrixXd img, Eigen::Tensor<double, 4> w_conv, Eigen::MatrixXd b_conv);

// function to get relu of given of convolution result
Eigen::Tensor<double, 3> relu_conv(Eigen::Tensor<double, 3> x);

// function to get 2x2 max pooling results for matrix
Eigen::Tensor<double, 3> pool2x2(Eigen::Tensor<double, 3> x);

// function to flatten the tensor and return a vector
Eigen::VectorXd flatten(Eigen::Tensor<double, 3> x);

// function to unflatten gradient vector to tensor
Eigen::Tensor<double, 3> flatten_backward(Eigen::VectorXd dl_dy, Eigen::Tensor<double, 3> x, Eigen::VectorXd y);

// function to get 2x2 max pooling backward
Eigen::Tensor<double, 3> pool2x2_backward(Eigen::Tensor<double, 3> dl_dy, Eigen::Tensor<double, 3> x, Eigen::Tensor<double, 3> y);

// function to get relu of given of convolution result
Eigen::Tensor<double, 3> relu_conv_backward(Eigen::Tensor<double, 3> dl_dy, Eigen::Tensor<double, 3> x, Eigen::Tensor<double, 3> y_pred);

// function to convolution backwards
void conv_backward(Eigen::Tensor<double, 3> dl_dy, Eigen::MatrixXd img, Eigen::Tensor<double, 4> w_conv, Eigen::MatrixXd b_conv, Eigen::Tensor<double, 3> y,
	Eigen::Tensor<double, 4>& dl_dw_conv, Eigen::MatrixXd& dl_db_conv);

