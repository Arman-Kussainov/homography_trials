// https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html

// 07.11.2019 Image threshold C/C++ OpenCV CUDA ...
// 30.07.2019 Cleaning the messy code


//#include <C:/opencv_320_vc14/opencv/build/include/opencv2/highgui/highgui.hpp>
//#include <C:/opencv_320_vc14/opencv/build/include/opencv2/imgproc/imgproc.hpp>
//#include "opencv2/imgcodecs.hpp"

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
//using namespace cv;
using namespace cv::xfeatures2d;


// simple function to handke the brightness of the image
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

	// to define image properites while writing it to disk
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	// load and modify first image to initialize proper Mat files
	Mat IMG;
	string p_ath = "C:\\adani_data_correction\\sample_2\\";

	// Adjust contrast (a_lpha) and brightness (b_eta)
	int a_lpha = 15;
	int b_eta = 30;

	IMG = imread("C:\\adani_data_correction\\sample_2\\000.tif", CV_LOAD_IMAGE_UNCHANGED);
	// Rotate these original images 90 ccw
	transpose(IMG, IMG); flip(IMG, IMG, 0);

	// cutting the strip from orioginal image for the messy corner detection procedure
	int strip_width = IMG.rows / 3.7;
	int strip_cut = strip_width / 1.1;
	int x_left = 1100;
	int x_right = x_left+1600;
	Mat strip_roi = IMG(Rect(0+x_left, IMG.rows - strip_width, IMG.cols - x_right, strip_width-strip_cut));

	// detected in images shift
	int horiz_shift = 0;

	// working through all available images
	for (int d_egree = 0; d_egree <= 260; d_egree += 2) {

		string counte_r = SSTR(d_egree);
		// image's name has padding zeros '0' in front of the main counter
		counte_r = std::string(3 - counte_r.length(), '0') + counte_r;
		string extensio_n = ".tif";

		string filenam_e = p_ath + counte_r + extensio_n;
		IMG = imread(filenam_e, CV_LOAD_IMAGE_UNCHANGED);

		// Rotate these original images 90 ccw
		transpose(IMG, IMG); flip(IMG, IMG, 0);
		printf("degree=%d\t%d\n", d_egree, horiz_shift);

		// Adjust contrast (a_lpha) and brightness (b_eta)
		int a_lpha = 15;
		int b_eta = 30;
		Mat new_image = Mat::zeros(IMG.size(), IMG.type());
		new_image = brightness_contrast_cv(IMG, a_lpha, b_eta);

		// cutting the strip from the original image to allign the sample holder accros all images
		Mat strip_roi = new_image(Rect(0 + x_left, IMG.rows - strip_width, IMG.cols - x_right, strip_width - strip_cut));
		strip_roi.convertTo(strip_roi, CV_32FC1, 1.0 / 255.0);

		// really messy function though producing some reasonable results
		Mat c_orners = Mat::zeros(strip_roi.size(), CV_32FC1);
		int blockSize = 10;
		int apertureSize = 21;
		double k = 0.04;
		cornerHarris(strip_roi, c_orners, blockSize, apertureSize, k, BORDER_DEFAULT);
		Mat c_orners_norm, c_orners_norm_scaled;
		normalize(c_orners, c_orners_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(c_orners_norm, c_orners_norm);

		// just copy pasted something from web to make a two value images without any grayscale
		double min=0, max=252;
		Mat M_ask;
		inRange(c_orners_norm, min, max, M_ask);
		Mat nonZeroCoordinates;
		findNonZero(M_ask, nonZeroCoordinates);

		// this loop is used to calculate the X position of the reference feature in the image 
		int ave_index = 0;
		for (int i = 0; i < nonZeroCoordinates.total(); i++) {
			ave_index += nonZeroCoordinates.at<Point>(i).x;
			//cout << "Zero#" << i << ": " << nonZeroCoordinates.at<Point>(i).x << ", " << nonZeroCoordinates.at<Point>(i).y << endl;
		}
		
		horiz_shift = ave_index / nonZeroCoordinates.total();
		
		// shift image and store it in imgTranslated
		Mat imgTranslated(new_image.size(), new_image.type(), cv::Scalar::all(0));
		new_image(Rect(horiz_shift, 0, new_image.cols - horiz_shift, new_image.rows)).
			copyTo(imgTranslated(Rect(0, 0, new_image.cols - horiz_shift, new_image.rows)));
		
		// cut the edges and all other crap from the processed image
		Mat processed_image = imgTranslated(Rect(650, 460,1450,1500));

		namedWindow("result", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("result", processed_image);
		waitKey(1);

		//	обработка исходного изображения и запись на диск
		string n_ame = "_slices_";
		string pat_h = "C:\\adani_data_correction\\sample_2_clean\\";
		string c_ounter = SSTR(d_egree);
		filenam_e = pat_h + c_ounter + ".png";

		// save the image
		// cv::imwrite(filenam_e, processed_image, compression_params);


		// make the loop goe on forever
		if (d_egree == 260) { d_egree = -2; }

	}
	
	return 0;
}
