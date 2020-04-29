#include <vector>
#include <Eigen/Core>
#include "utils.h"
#include "nn_utils.h"
using namespace std;

class SingleLayerPerceptron{
	private:
		Eigen::MatrixXd w,b;

	public:
		// function to train single layer perceptron
		void train(vector<vector<Eigen::VectorXd>>& mini_x, vector<vector<Eigen::VectorXd>>& mini_y){
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
			for(int iter = 1; iter<15000; iter++){
				if(iter%2000 == 0) lr = lr*decay;
				Eigen::MatrixXd dL_dw = Eigen::MatrixXd::Zero(w.rows(), w.cols());
				Eigen::MatrixXd dL_db = Eigen::MatrixXd::Zero(b.rows(), b.cols());

				// iterate over current batch;
				for(int i=0;i<mini_x[k].size();i++){
					Eigen::VectorXd y_tilde = fc(mini_x[k][i], w, b);
					double loss=0.0;
					Eigen::VectorXd dl_dy;

					get_cross_entropy_loss(y_tilde, mini_y[k][i], loss, dl_dy);

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

				// function to test single layer perceptron linear
		void test(vector<Eigen::VectorXd> test_data, vector<int> test_labels){
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
};