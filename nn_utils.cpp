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

// function to get relu of given matrix
Eigen::VectorXd relu(Eigen::VectorXd x){
	for(int i=0;i<x.size();i++){
		x[i] = max(0.0, x[i]);
	}
	return x;
}

// function to get relu backward
Eigen::VectorXd relu_backwards(Eigen::VectorXd dl_dy, Eigen::VectorXd x){
	for(int i=0;i<x.size();i++){
		if(x[i] <= 0.0) dl_dy[i] = 0.0;
	}
	return dl_dy;
}

// function to convolve over given image
Eigen::Tensor conv(Eigen::MatrixXd img, Eigen::Tensor<double, 4> w_conv, Eigen::MatrixXd b_conv){
	Eigen::Tensor<double, 3> y(img.rows(), img.cols(), w_conv.dimension(3));
	y.setZero();

	
}

// function to get relu of given of convolution result
Eigen::Tensor relu_conv(Eigen::Tensor x){
	for(int i=0;i<x.dimension(0);i++){
		for(int j=0;j<x.dimension(1);j++){
			for(int k=0;k<x.dimension(2);k++){
				x(i,j,k) = max(0.0, x(i,j,k));
			}
		}
	}
	return x;
}


// function to get relu of given of convolution result
Eigen::Tensor relu__conv_backward(Eigen::Tensor dl_dy, Eigen::Tensor x, Eigen::Tensor y_pred){
	for(int i=0;i<x.dimension(0);i++){
		for(int j=0;j<x.dimension(1);j++){
			for(int k=0;k<x.dimension(2);k++){
				if(x(i,j,k) <= 0.0) dl_dy(i,j,k) = 0.0;
			}
		}
	}
	return dl_dy;
}

// function to get 2x2 max pooling results for matrix
Eigen::Tensor pool2x2(Eigen::MatrixXd x){
	Eigen::Tensor<double, 3> y(x.dimension(0)/2, x.dimension(1)/2, x.dimension(3));
	y.setZero();
	int a = 0;
	int b = 0;
	for(int i=0;i<x.dimension(0);i+=2){
		for(int j=0;j<x.dimension(1);j+=2){
			for(int k=0;k<x.dimension(2);k++){
				y(a,b,k) = max(max(x(i,j,k), x(i+1,j,k)), max(x(i,j+1,k), x(i+1,j+1,k)));
			}
			b++;
		}
		a++;
		b=0;
	}
	return y;
}

// function to get 2x2 max pooling backward
Eigen::Tensor pool2x2_backward(Eigen::Tensor dl_dy, Eigen::Tensor x, Eigen::Tensor y){
	Eigen::Tensor<double, 3> result(x.dimension(0), x.dimension(1), x.dimension(3));
	result.setZero();
	int a = 0;
	int b = 0;
	for(int i=0;i<x.dimension(0);i+=2){
		for(int j=0;j<x.dimension(1);j+=2){
			for(int k=0;k<x.dimension(2);k++){
				double max_val = max(max(x(i,j,k), x(i+1,j,k)), max(x(i,j+1,k), x(i+1,j+1,k)));
				if(max_val == x(i,j,k)) result(i,j,k) = dl_dy(a,b,k);
				if(max_val == x(i+1,j,k)) result(i+1,j,k) = dl_dy(a,b,k);
				if(max_val == x(i,j+1,k)) result(i,j+1,k) = dl_dy(a,b,k);
				if(max_val == x(i+1,j+1,k)) result(i+1,j+1,k) = dl_dy(a,b,k);
			}
			b++;
		}
		a++;
		b=0;
	}
	return result;	
}

