#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils.h"
#include "nn_utils.h"
using namespace std;

class ConvolutionalNeuralNetwork{
	private:
		Eigen::MatrixXd w_fc,b_fc, b_conv;
		Eigen::Tensor<double, 4> w_conv(3,3,1,3);

		// function to intialize tensor
		Eigen::Tensor init_tensor(double low, double high, Eigen::Tensor t){
			
			double range = high - low;

			t.setRandom();
			Eigen::Tensor<double, 4> temp(3,3,1,3);
			temp.setConstant(0.0);
			t = (t + temp)*range/2.0;

			temp.setConstant(low);
			t = (t + temp);

			return t;
		}

		Map<mat> reshape (vec& b, const uint n, const uint m) {
		    return Map<const mat>(b.data(), n, m);
		}

	public:
		// function to train single layer perceptron
		void train(vector<vector<Eigen::VectorXd>>& mini_x, vector<vector<Eigen::VectorXd>>& mini_y){
			double lr = 0.1;
			double decay = 0.9;

			// initialize weights
			w_conv = init_tensor(0.0, 0.1, w_conv);
			w_fc = get_random_matrix(0.0, 0.1, 10, 147);
			
			b_conv = get_random_matrix(0.0, 0.1, 3, 1);
			b_fc = get_random_matrix(0.0, 0.1, 10, 1);

			// batch number and batch, epoch loss;
			int k=0;
			double batch_loss = 0.0;
			double epoch_loss = 0.0;

			cout<<"Training Convolutional Neural Network"<<endl;
			for(int iter = 1; iter<20000; iter++){
				if(iter%2000 == 0) lr = lr*decay;

				Eigen::Tensor<double, 4> dL_dw_conv = w_conv.constant(0.0);
				Eigen::MatrixXd dL_dw_fc = Eigen::MatrixXd::Zero(w_fc.rows(), w_fc.cols());

				Eigen::MatrixXd dL_db_conv = Eigen::MatrixXd::Zero(b_conv.rows(), b_conv.cols());
				Eigen::MatrixXd dL_db_fc = Eigen::MatrixXd::Zero(b_fc.rows(), b_fc.cols());

				// iterate over current batch;
				for(int i=0;i<mini_x[k].size();i++){

					Eigen::MatrixXd img = reshape(mini_x[k][i], 14, 14);

					Eigen::Tensor pred1 = conv(img, w_conv, b_conv);
					Eigen::Tensor pred2 = relu_conv(pred1);
					Eigen::Tensor pred3 = pool2x2(pred2);
					Eigen::VectorXd pred4 = flatten(pred3);
					Eigen::VectorXd y_tilde = fc(pred4, w_fc, b_fc);
					
					double loss=0.0;
					Eigen::VectorXd dl_dy;
					get_cross_entropy_loss(y_tilde2, mini_y[k][i], loss, dl_dy);

					batch_loss += loss;

					// initialize gradients

					Eigen::MatrixXd dl_dx_fc = Eigen::MatrixXd::Zero(pred4.rows(), pred4.cols());
					Eigen::MatrixXd dl_dw_fc = Eigen::MatrixXd::Zero(w_fc.rows(), w_fc.cols());
					Eigen::MatrixXd dl_db_fc = Eigen::MatrixXd::Zero(b_fc.rows(), b_fc.cols());

					// backpropogate gradients
					fc_backward(dl_dy, activation, w2, b2, y_tilde2,
						dl_dx_fc, dl_dw_fc, dl_db_fc);

					Eigen::Tensor dl_dx = flatten_backward(dl_dx_fc, pred3, pred4);
					Eigen::Tensor dl_dx = pool2x2_backward(dl_dx, pred2, pred3);
					Eigen::Tensor dl_dx = relu__conv_backward(dl_dx, pred1, pred2);

					conv_backward(dl_dx, img, w_conv, b_conv, pred1,
						dl_dw_conv, dl_db_conv);

					// add the gradients for this batch
					dL_dw_conv = dL_dw_conv + dl_dw_conv;
					dL_db_conv = dL_db_conv + dl_db_conv;

					dL_dw_fc = dL_dw_fc + dl_dw_fc;
					dL_db_fc = dL_db_fc + dl_db_fc;
				}
				// update weights
				w_conv = w_conv - dL_dw_conv*(lr/mini_x[k].size());
				b_conv = b_conv - dL_db_conv*(lr/mini_x[k].size());

				w_fc = w_fc - dL_dw_fc*(lr/mini_x[k].size());
				b_fc = b_fc - dL_db_fc*(lr/mini_x[k].size());

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

				// function to test single layer perceptron linear
		void test(vector<Eigen::VectorXd> test_data, vector<int> test_labels){
			double accuracy = 0.0;
			cout<<"Testing Multi Layer Perceptron"<<endl;
			for(int i=0;i<test_data.size();i++){
				
				// forward propogation in network
				Eigen::MatrixXd img = reshape(mini_x[k][i], 14, 14);

				Eigen::MatrixXd pred1 = conv(img, w_conv, b_conv);
				Eigen::MatrixXd pred2 = relu_conv(pred1);
				Eigen::MatrixXd pred3 = pool2x2(pred2);
				Eigen::VectorXd pred4 = flatten(pred3);
				Eigen::VectorXd y_predict_label = fc(pred4, w_fc, b_fc);
				
				// get index of max value
				int y_predict_label;
				y_predict.maxCoeff(&y_predict_label);

				if(y_predict_label == test_labels[i]) accuracy+=1.0;
			}

			cout<<"Model Accuracy: "<<(accuracy/test_data.size())*100<<endl;
			return;
		}
};