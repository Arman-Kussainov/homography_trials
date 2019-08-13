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
	Mat IMG;
	string p_ath = "C:\\0_adani_data_correction\\sample_2\\";

	int d_egree = 0;
	string counte_r = SSTR(d_egree);
	counte_r = std::string(3 - counte_r.length(), '0') + counte_r;
	string extensio_n = ".tif";

	string filenam_e = p_ath + counte_r + extensio_n;
	IMG = imread(filenam_e, CV_LOAD_IMAGE_UNCHANGED);

	// Rotate these original images 90 ccw
	transpose(IMG, IMG); flip(IMG, IMG, 0);
	//printf("columns=%d rows=%d\n", IMG.cols, IMG.rows);
	printf("degree=%d\n", d_egree);

	// Adjust contrast (a_lpha) and brightness (b_eta)
	int a_lpha = 15;
	int b_eta = 30;
	Mat new_image = Mat::zeros(IMG.size(), IMG.type());
	new_image = brightness_contrast_cv(IMG, a_lpha, b_eta);

	Mat im_src=new_image;
	vector<Point2f> pts_src;

	pts_src.push_back(Point2f(1420, 1819));
	pts_src.push_back(Point2f(1907, 1828));
	pts_src.push_back(Point2f(559, 1991));
	//pts_src.push_back(Point2f(2477, 1950));

	d_egree = 260;
	counte_r = SSTR(d_egree);
	counte_r = std::string(3 - counte_r.length(), '0') + counte_r;
	extensio_n = ".tif";

	filenam_e = p_ath + counte_r + extensio_n;
	IMG = imread(filenam_e, CV_LOAD_IMAGE_UNCHANGED);

	// Rotate these original images 90 ccw
	transpose(IMG, IMG); flip(IMG, IMG, 0);
	//printf("columns=%d rows=%d\n", IMG.cols, IMG.rows);
	printf("degree=%d\n", d_egree);

	//double min, max;
	//minMaxLoc(IMG, &min, &max);

	// Adjust contrast (a_lpha) and brightness (b_eta)
	new_image = brightness_contrast_cv(IMG, a_lpha, b_eta);

	Mat im_dst = new_image;
	vector<Point2f> pts_dst;

	pts_dst.push_back(Point2f(1212, 1824));
	pts_dst.push_back(Point2f(1700, 1828));
	pts_dst.push_back(Point2f(364, 1996));
	//pts_dst.push_back(Point2f(2284, 1955));

	Mat h = getAffineTransform(pts_src, pts_dst);


		//		double min, max;
		//		minMaxLoc(new_image, &min, &max);
		//		new_image = abs(new_image - max);
		//		min = 7000;
		//		max = 60535;
		//		Mat M_ask;
		//		inRange(new_image, min, max, M_ask);
		//		M_ask.convertTo(M_ask, CV_16UC1, 1.f / 255);
		//		new_image = new_image.mul(M_ask);

		// Output image
	
	Mat im_out;
	// Warp source image to destination based on homography
	warpAffine(im_src, im_out, h, im_dst.size());
	
	Mat s_um = im_src + im_dst;
	namedWindow("Dinara", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	imshow("Dinara", s_um);



	// Display images
	//namedWindow("Source Image", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	//imshow("Source Image", im_src);
	//namedWindow("Destination Imag", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	//imshow("Destination Image", im_dst);
	//namedWindow("Warped Source Image", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	//imshow("Warped Source Image", im_out);

	waitKey(0);

		//	slice_all=slice_all.mul(M_ask);
		//	slice_all = slice_all / 3000.0 * 0.2;
		//	slice_all.setTo(1.0, slice_all > .7);

		// To make it run forever
		// if (d_egree == 260) { d_egree = -2; }
	
	return 0;
}