#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

// see https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
void alignImages(Mat &im1, Mat &im2, Mat &im1_1, Mat &im2_1, Mat &im1Reg, Mat &h)

{

	// Convert images to grayscale
	Mat im1Gray=im1, im2Gray=im2;
	//cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	//cvtColor(im2, im2Gray, COLOR_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());


	// Draw top matches
	Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	namedWindow("imMatches", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	imshow("imMatches", imMatches);


	// Extract location of good matches
	std::vector<Point2f> points1, points2;
	cout << matches.size();
	// affine transform need only 3 points... need to use anohter ones
	for (size_t i = 0; i < matches.size(); i++)
	//for (size_t i = 0; i < 3; i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
		cout<<keypoints1[i].pt << "\t"<< keypoints2[i].pt << "\n";
	}

	// Find homography
	h = findHomography(points1, points2, RANSAC);
	//h = getAffineTransform(points1, points2);

	// Use homography to warp image
	warpPerspective(im1, im1Reg, h, im2.size());
	//warpAffine(im1, im1Reg, h, im2.size());

}

cv::Mat brightness_contrast_cv(cv::Mat IMG, int a_lpha, int b_eta, int strip) {
	// change brightness and contrast
	cv::Mat my_image = cv::Mat::zeros(IMG.size(), IMG.type());
	my_image = a_lpha*IMG + b_eta;
	// invert the image
	transpose(my_image, my_image); flip(my_image, my_image, 0);
	// cut the proper region
	
	cv::Mat strip_roi;
	cv::Mat new_mat;
	if (strip == 0) {
		int x_upper = 800;
		int y_upper = 450;
		int widt_h = 1500;
		int heigh_t = 1500;
		strip_roi = my_image(Rect(x_upper, y_upper, widt_h, heigh_t));
		new_mat = strip_roi;
	}
	if (strip == 1) {
		int x_upper = 800;
		int y_upper = 1850;
		int widt_h = 1500;
		int heigh_t = 100;

		strip_roi = my_image(Rect(x_upper, y_upper, widt_h, heigh_t));

		Mat imgTranslated(my_image.size(), my_image.type(), cv::Scalar::all(255));
		strip_roi(Rect(0, 0, strip_roi.cols, strip_roi.rows)).
			copyTo(imgTranslated(Rect(x_upper, y_upper, widt_h, heigh_t)));
		new_mat = imgTranslated;
		
	}
	return new_mat;
}

int main(int argc, char **argv)
{
	// Adjust contrast (a_lpha) and brightness (b_eta)
	int a_lpha = 15;
	int b_eta = 30;

	// Read reference image
	string refFilename("C:\\adani_data_correction\\sample_2\\030.tif");
	cout << "Reading reference image : " << refFilename << endl;
	Mat imReference = imread(refFilename, IMREAD_GRAYSCALE);

	// Adjust contrast (a_lpha) and brightness (b_eta) and rotate
	Mat im1_1 = brightness_contrast_cv(imReference, a_lpha, b_eta,1);
	Mat im1 = brightness_contrast_cv(imReference, a_lpha, b_eta, 0);


	// Read image to be aligned
	string imFilename("C:\\adani_data_correction\\sample_2\\032.tif");
	cout << "Reading image to align : " << imFilename << endl;

	Mat im = imread(imFilename, IMREAD_GRAYSCALE);
	// Adjust contrast (a_lpha) and brightness (b_eta) and rotate
	Mat im2_1 = brightness_contrast_cv(im, a_lpha, b_eta,1);
	Mat im2 = brightness_contrast_cv(im, a_lpha, b_eta, 0);


	// Registered image will be resotred in imReg. 
	// The estimated homography will be stored in h. 
	Mat imReg, h;

	// Align images
	cout << "Aligning images ..." << endl;
	alignImages(im2, im1, im2_1, im1_1, imReg, h);

	// Write aligned image to disk. 
	string outFilename("aligned.jpg");
	cout << "Saving aligned image : " << outFilename << endl;
	imwrite(outFilename, imReg);


	// Show final result
	namedWindow("imReference", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	imshow("imReference", im1);

	namedWindow("Im", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	imshow("Im", im2);

	namedWindow("imReg", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	imshow("imReg", imReg);

	waitKey(0);

	// Print estimated homography
	cout << "Estimated homography : \n" << h << endl;
}












/*#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <opencv2/video/tracking.hpp>
#include <ctime>
using namespace cv;
using namespace cv::xfeatures2d;


// simple function to handle the brightness of the image
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
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	// load and modify first image to initialize proper Mat files
	Mat IMG;
	string p_ath = "C:\\adani_data_correction\\sample_2\\";

	IMG = imread("C:\\adani_data_correction\\sample_2\\000.tif", IMREAD_UNCHANGED);
	// Rotate these original images 90 ccw
	transpose(IMG, IMG); flip(IMG, IMG, 0);

	// Adjust contrast (a_lpha) and brightness (b_eta)
	int a_lpha = 15;
	int b_eta = 30;

	// Adjust contrast (a_lpha) and brightness (b_eta)
	Mat new_image = Mat::zeros(IMG.size(), IMG.type());
	new_image = brightness_contrast_cv(IMG, a_lpha, b_eta);
	
	// detected in images shift
	int horiz_shift = 318;

	// shift image and store it in imgTranslated
	Mat imgTranslated(new_image.size(), new_image.type(), cv::Scalar::all(0));
	new_image(Rect(horiz_shift, 0, new_image.cols - horiz_shift, new_image.rows)).
		copyTo(imgTranslated(Rect(0, 0, new_image.cols - horiz_shift, new_image.rows)));

	int Tpx= 650, Tpy= 460, Tpwidth= 1450, Tpheight= 1450;
	// cut the edges and all other crap from the processed image
	Mat processed_image = imgTranslated(Rect(Tpx, Tpy, Tpwidth, Tpheight));

	// pair of images to work with
	Mat im1 = processed_image;
	//Mat im1 = imread("C:\\adani_data_correction\\sample_2_clean\\0.png", IMREAD_UNCHANGED);

	// cutting the strip from original image for the messy corner detection procedure
	int strip_width = IMG.rows / 3.7;
	int strip_cut = strip_width / 1.1;
	int x_left = 1100;
	int x_right = x_left+1600;
	Mat strip_roi = IMG(Rect(0+x_left, IMG.rows - strip_width, IMG.cols - x_right, strip_width-strip_cut));

	horiz_shift = 0;
	p_ath = "C:\\adani_data_correction\\sample_2_clean\\";
	// working through all available images
	for (int d_egree = 2; d_egree <= 260; d_egree += 2) {
		clock_t begin = clock();
		string counte_r = SSTR(d_egree);
		// image's name has padding zeros '0' in front of the main counter
		//counte_r = std::string(3 - counte_r.length(), '0') + counte_r;
		string extensio_n = ".png";

		string filenam_e = p_ath + counte_r + extensio_n;
		printf("%d\t", d_egree);
		Mat im2 = imread(filenam_e, IMREAD_UNCHANGED);

		// make the loop goes on forever
		// if (d_egree == 260) { d_egree = -2; }

		// Convert images to gray scale; ?????????????????????????????
		Mat im1_gray, im2_gray;
		im1.convertTo(im1_gray, CV_32F);
		im2.convertTo(im2_gray, CV_32F);

		// Define the motion model
		const int warp_mode = MOTION_HOMOGRAPHY;

		// Set a 2x3 or 3x3 warp matrix depending on the motion model.
		Mat warp_matrix;

		// Initialize the matrix to identity
		if (warp_mode == MOTION_HOMOGRAPHY)
			warp_matrix = Mat::eye(3, 3, CV_32F);
		else
			warp_matrix = Mat::eye(2, 3, CV_32F);

		// Specify the number of iterations.
		int number_of_iterations = 5000;

		// Specify the threshold of the increment
		// in the correlation coefficient between two iterations
		double termination_eps = 1e-10;

		// Define termination criteria
		TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);

		// Run the ECC algorithm. The results are stored in warp_matrix.
		findTransformECC(
			im1_gray,
			im2_gray,
			warp_matrix,
			warp_mode,
			criteria
		);

		// Storage for warped image.
		Mat im2_aligned;

		if (warp_mode != MOTION_HOMOGRAPHY)
		// Use warpAffine for Translation, Euclidean and Affine
			warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
		else
		// Use warpPerspective for Homography
		warpPerspective(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);

		// Show final result
		namedWindow("Image 1", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("Image 1", im1);

		namedWindow("Image 2", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("Image 2", im2);

		namedWindow("Image 2 Aligned", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("Image 2 Aligned", im2_aligned);

		waitKey(1);
		
		im1 = im2;

		//	обработка исходного изображения и запись на диск
		string n_ame = "_slices_";
		string pat_h = "C:\\adani_data_correction\\sample_2_homography\\";
		string c_ounter = SSTR(d_egree);
		filenam_e = pat_h + c_ounter + ".png";

		// save the image
		cv::imwrite(filenam_e, im2_aligned, compression_params);
		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		printf("%f\n", elapsed_secs);


	}
	return 0;
}
*/