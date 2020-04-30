main: main.cpp utils.h nn_utils.h
	g++ -w -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -I/usr/local/include/eigen3/ -g utils.h utils.cpp nn_utils.h nn_utils.cpp SLPLinear.cpp SingleLayerPerceptron.cpp MultiLayerPerceptron.cpp main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching

clean:
	rm -f cnn.out
	rm -f a.out
	rm -f utils.h.gch
	rm -f nn_utils.h.gch