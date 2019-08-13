// bone_images.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// 07.11.2019 Image threshold C/C++ OpenCV CUDA ...

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <C:/opencv_320_vc14/opencv/build/include/opencv2/highgui/highgui.hpp>
#include <C:/opencv_320_vc14/opencv/build/include/opencv2/imgproc/imgproc.hpp>
#include <C:/opencv_320_vc14/opencv/build/include/opencv2/core/core.hpp>
#include <C:/opencv_320_vc14/opencv/build/include/opencv2/core/cuda.hpp>

#include "opencv2/calib3d/calib3d.hpp"

cv::Mat brightness_contrast_cv(cv::Mat IMG, int a_lpha, int b_eta) {
	cv::Mat new_image = cv::Mat::zeros(IMG.size(), IMG.type());
	new_image = a_lpha*IMG + b_eta;
	return new_image;
}

using namespace cv;
using namespace std;

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char* argv[])
{

	// Adjust contrast (a_lpha) and brightness (b_eta)
	int a_lpha = 15;
	int b_eta = 30;

	Mat IMG;
	string p_ath = "C:\\0_adani_data_correction\\sample_2\\";
	IMG = imread("C:\\0_adani_data_correction\\sample_2\\000.tif", CV_LOAD_IMAGE_UNCHANGED);
	// Rotate these original images 90 ccw
	transpose(IMG, IMG); flip(IMG, IMG, 0);

	Mat previous_image = Mat::zeros(IMG.size(), IMG.type());
	Mat current_image = Mat::zeros(IMG.size(), IMG.type());

	for (int d_egree = 0; d_egree <= 260; d_egree+=2) {
		string counte_r = SSTR(d_egree);
		counte_r = std::string(3 - counte_r.length(), '0') + counte_r;
		string extensio_n = ".tif";

		string filenam_e = p_ath + counte_r + extensio_n;
		IMG = imread(filenam_e, CV_LOAD_IMAGE_UNCHANGED);

		// Rotate these original images 90 ccw
		transpose(IMG, IMG); flip(IMG, IMG, 0);
		//printf("columns=%d rows=%d\n", IMG.cols, IMG.rows);
		printf("degree=%d\n", d_egree);

		Mat new_image = Mat::zeros(IMG.size(), IMG.type());
		new_image = brightness_contrast_cv(IMG, a_lpha, b_eta);
		
		// blending two images
		double alpha = 0.8; double beta;
		beta = (1.0 - alpha);
		addWeighted(new_image, alpha, previous_image, beta, 0.0, current_image);
		//previous_image = current_image;
		previous_image += (new_image/60);

		//int strip_width = new_image.rows / 4;
		//Mat bottom_strip = new_image(Rect(0, new_image.rows - strip_width, new_image.cols, strip_width));
		//bottom_strip = brightness_contrast_cv(bottom_strip, 10, 10000);

		namedWindow("result", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("result", previous_image);
		waitKey(1);

		// To make it run forever
		//if (d_egree == 260) { d_egree = -2; }
	}
	waitKey(0);
	return 0;
}