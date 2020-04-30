#include <vector>
#include <Eigen/Core>
#include "utils.h"
#include "nn_utils.h"
using namespace std;

class MultiLayerPerceptron{
	private:
		Eigen::MatrixXd w1,w2,b1,b2;

	public:
		// function to train single layer perceptron
		void train(vector<vector<Eigen::VectorXd>>& mini_x, vector<vector<Eigen::VectorXd>>& mini_y){
			double lr = 0.1;
			double decay = 0.9;

			// initialize weights
			w1 = get_random_matrix(0.0, 0.1, 30, mini_x[0][0].size());
			w2 = get_random_matrix(0.0, 0.1, 10, 30);
			
			b1 = get_random_matrix(0.0, 0.1, 30, 1);
			b2 = get_random_matrix(0.0, 0.1, 10, 1);

			// batch number and batch, epoch loss;
			int k=0;
			double batch_loss = 0.0;
			double epoch_loss = 0.0;

			cout<<"Training Multi Layer Perceptron"<<endl;
			for(int iter = 1; iter<20000; iter++){
				if(iter%2000 == 0) lr = lr*decay;
				Eigen::MatrixXd dL_dw1 = Eigen::MatrixXd::Zero(w1.rows(), w1.cols());
				Eigen::MatrixXd dL_dw2 = Eigen::MatrixXd::Zero(w2.rows(), w2.cols());

				Eigen::MatrixXd dL_db1 = Eigen::MatrixXd::Zero(b1.rows(), b1.cols());
				Eigen::MatrixXd dL_db2 = Eigen::MatrixXd::Zero(b2.rows(), b2.cols());

				// iterate over current batch;
				for(int i=0;i<mini_x[k].size();i++){
					Eigen::VectorXd y_tilde1 = fc(mini_x[k][i], w1, b1);
					
					Eigen::VectorXd activation = relu(y_tilde1);

					Eigen::VectorXd y_tilde2 = fc(activation, w2, b2);
					
					double loss=0.0;
					Eigen::VectorXd dl_dy;
					get_cross_entropy_loss(y_tilde2, mini_y[k][i], loss, dl_dy);

					batch_loss += loss;

					// initialize gradients
					Eigen::MatrixXd dl_dx1 = Eigen::MatrixXd::Zero(mini_x[k][i].rows(), mini_x[k][i].cols());
					Eigen::MatrixXd dl_dw1 = Eigen::MatrixXd::Zero(w1.rows(), w1.cols());
					Eigen::MatrixXd dl_db1 = Eigen::MatrixXd::Zero(b1.rows(), b1.cols());

					Eigen::MatrixXd dl_dx2 = Eigen::MatrixXd::Zero(activation.rows(), activation.cols());
					Eigen::MatrixXd dl_dw2 = Eigen::MatrixXd::Zero(w2.rows(), w2.cols());
					Eigen::MatrixXd dl_db2 = Eigen::MatrixXd::Zero(b2.rows(), b2.cols());



					// backpropogate gradients
					fc_backward(dl_dy, activation, w2, b2, y_tilde2,
						dl_dx2, dl_dw2, dl_db2);

					Eigen::VectorXd dl_da = relu_backwards(dl_dx2, y_tilde1);

					fc_backward(dl_da, mini_x[k][i], w1, b1, y_tilde1,
						dl_dx1, dl_dw1, dl_db1);

					// add the gradients for this batch
					dL_dw1 = dL_dw1 + dl_dw1;
					dL_db1 = dL_db1 + dl_db1;

					dL_dw2 = dL_dw2 + dl_dw2;
					dL_db2 = dL_db2 + dl_db2;
				}
				// update weights
				w1 = w1 - dL_dw1*(lr/mini_x[k].size());
				b1 = b1 - dL_db1*(lr/mini_x[k].size());

				w2 = w2 - dL_dw2*(lr/mini_x[k].size());
				b2 = b2 - dL_db2*(lr/mini_x[k].size());

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
				Eigen::VectorXd y_tilde1 = fc(test_data[i], w1, b1);

				Eigen::MatrixXd activation = relu(y_tilde1);

				Eigen::VectorXd y_predict = fc(activation, w2, b2);
				
				// get index of max value
				int y_predict_label;
				y_predict.maxCoeff(&y_predict_label);

				if(y_predict_label == test_labels[i]) accuracy+=1.0;
			}

			cout<<"Model Accuracy: "<<(accuracy/test_data.size())*100<<endl;
			return;
		}
};