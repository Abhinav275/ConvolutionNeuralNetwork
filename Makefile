main: main.cpp utils.h nn_utils.h
	g++ -o main -w -L/usr/local/lib/ -I/usr/local/include/eigen3/ -g utils.h utils.cpp nn_utils.h nn_utils.cpp SLPLinear.cpp SingleLayerPerceptron.cpp MultiLayerPerceptron.cpp ConvolutionalNeuralNetwork.cpp main.cpp

clean:
	rm -f main.out
	rm -f a.out
	rm -f utils.h.gch
	rm -f nn_utils.h.gch