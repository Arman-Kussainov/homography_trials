// bone_images.cpp : Defines the entry point for the console application.
// https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html


#include "stdafx.h"

// 07.11.2019 Image threshold C/C++ OpenCV CUDA ...

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

#include <C:/opencv_320_vc14/opencv/build/include/opencv2/highgui/highgui.hpp>
#include <C:/opencv_320_vc14/opencv/build/include/opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

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
	Mat IMG;
	string p_ath = "C:\\0_adani_data_correction\\sample_2\\";

	// Adjust contrast (a_lpha) and brightness (b_eta)
	int a_lpha = 15;
	int b_eta = 30;

	IMG = imread("C:\\0_adani_data_correction\\sample_2\\000.tif", CV_LOAD_IMAGE_UNCHANGED);
	// Rotate these original images 90 ccw
	transpose(IMG, IMG); flip(IMG, IMG, 0);

	int strip_width = IMG.rows / 3.7;
	int strip_cut = strip_width / 1.1;
	int x_left = 1100;
	int x_right = x_left+1600;

	Mat strip_roi = IMG(Rect(0+x_left, IMG.rows - strip_width, IMG.cols - x_right, strip_width-strip_cut));

	Mat previous_image = Mat::zeros(strip_roi.size(), CV_32FC1);
	Mat current_image = Mat::zeros(strip_roi.size(), CV_32FC1);

	int horiz_shift = 0;
	for (int d_egree = 0; d_egree <= 260; d_egree += 2) {
		string counte_r = SSTR(d_egree);
		counte_r = std::string(3 - counte_r.length(), '0') + counte_r;
		string extensio_n = ".tif";

		string filenam_e = p_ath + counte_r + extensio_n;
		IMG = imread(filenam_e, CV_LOAD_IMAGE_UNCHANGED);

		// Rotate these original images 90 ccw
		transpose(IMG, IMG); flip(IMG, IMG, 0);
		//printf("columns=%d rows=%d\n", IMG.cols, IMG.rows);
		printf("degree=%d\t%d\n", d_egree, horiz_shift);

		// Adjust contrast (a_lpha) and brightness (b_eta)
		int a_lpha = 15;
		int b_eta = 30;
		Mat new_image = Mat::zeros(IMG.size(), IMG.type());
		new_image = brightness_contrast_cv(IMG, a_lpha, b_eta);

		Mat another_image = Mat::zeros(new_image.size(), new_image.type());
		another_image = new_image;
		Mat strip_roi = another_image(Rect(0 + x_left, IMG.rows - strip_width, IMG.cols - x_right, strip_width - strip_cut));
		strip_roi.convertTo(strip_roi, CV_32FC1, 1.0 / 255.0);

		Mat c_orners = Mat::zeros(strip_roi.size(), CV_32FC1);
		int blockSize = 10;
		int apertureSize = 21;
		double k = 0.04;
		cornerHarris(strip_roi, c_orners, blockSize, apertureSize, k, BORDER_DEFAULT);
		Mat c_orners_norm, c_orners_norm_scaled;
		normalize(c_orners, c_orners_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(c_orners_norm, c_orners_norm);

		double min=0, max=252;
		//minMaxLoc(c_orners_norm, &min, &max);
		Mat M_ask;
		inRange(c_orners_norm, min, max, M_ask);
//		M_ask.convertTo(M_ask, CV_16UC1, 1.f / 255);
		
		Mat nonZeroCoordinates;
		findNonZero(M_ask, nonZeroCoordinates);

		int ave_index = 0;
		for (int i = 0; i < nonZeroCoordinates.total(); i++) {
			ave_index += nonZeroCoordinates.at<Point>(i).x;
			//cout << "Zero#" << i << ": " << nonZeroCoordinates.at<Point>(i).x << ", " << nonZeroCoordinates.at<Point>(i).y << endl;
		}
		
		//if (d_egree>0) {
			horiz_shift = ave_index / nonZeroCoordinates.total();
		//}


		//cv::Mat img = cv::imread("image.jpg");
		Mat imgTranslated(new_image.size(), new_image.type(), cv::Scalar::all(0));

		   new_image(Rect(horiz_shift, 0, new_image.cols - horiz_shift, new_image.rows)).
copyTo(imgTranslated(Rect(0, 0, new_image.cols - horiz_shift, new_image.rows)));

		namedWindow("result", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("result", imgTranslated);
		waitKey(10);
		if (d_egree == 260) { d_egree = -2; }


		//cv::Mat img = cv::imread("image.jpg");
		//cv::Mat imgTranslated(img.size(), img.type(), cv::Scalar::all(0));
		//img(cv::Rect(50, 30, img.cols - 50, img.rows - 30)).copyTo(imgTranslated(cv::Rect(0, 0, img.cols - 50, img.rows - 30)));

	}
	
	return 0;
}
