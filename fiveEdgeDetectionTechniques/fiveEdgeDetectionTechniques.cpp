//0. import working image and libraries; declare variables, arrays and function prototypes to be used
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> //For Sobel and Scharr
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
Mat original = imread("E:/normandy.jpg", 1); //input the working image
char* window_name_4 = "Source image";
//Sobel combines Gaussian smoothing & differentation; Sobel and Scharr are almost the same, except they use different kernels, wher +/-1 is replaced by +/-3 & +/-2 by +/-10, see openCV 3.2.0 documentation
//Sobel & Scharr prototypes
Mat src, src_gray, grad, grad2;
Mat grad_x, grad_y, grad_x2, grad_y2;
Mat abs_grad_x, abs_grad_y, abs_grad_x2, abs_grad_y2;
const char* window_name = "Sobel Edge Detector";
const char* window_name_2 = "Sharr Edge Detector";
int scale = 1, delta = 0, ddepth = CV_16S;
//Canny prototypes
Mat dst, detected_edges;
int edgeThresh = 1, lowThreshold, max_lowThreshold = 100, ratio = 3, kernel_size = 3;
char* window_name_3 = "Canny Edge Detector";
void CannyThreshold(int, void*);
//Lapalcian method finds the sum of the second derivatives of pixel graylevel intensities in x and in y; here we use a Gaussian mask before we apply the Laplacian
//Laplacian prototypes
Mat smoothed, laplace, result, finalLaplace;
enum { GAUSSIAN, BLUR, MEDIAN }; //enum is a data type made up a of set of constants (known as numerators); here we are storing smoothing types
int sigma = 3, smoothType = GAUSSIAN; //can be GAUSSIAN, BLUR or MEDIAN (blur); Since a Gaussian mask is applied, the method bexomes what is a known as "Laplacian-of-Gaussian"
char* window_Laplacian = "Laplacian operator";
//Harris prototypes
Mat dst2, dst_norm, dst_norm_scaled;
int thresh = 150, max_thresh = 255, blockSize = 2, apertureSize = 3; 
double k = 0.04; // Detector parameters
char* corners_window = "Harris Corner Detector";
void cornerHarris_demo(int, void*);
//Main function
int main(int argc, char** argv) {
	resize(original, src, Size(0.32*original.cols, 0.32*original.rows), 0, 0, INTER_LINEAR); //resize the working image and store it in "src"
	imshow(window_name_4, src); //Show unprocessed, original image
	cvtColor(src, src_gray, COLOR_BGR2GRAY); // Convert the image to grayscale so we can perform edge detection on it
	for (;;) { //Infinite loop
		//Harris Implementation
		createTrackbar("Threshold: ", corners_window, &thresh, max_thresh, cornerHarris_demo); //Haris trackbar
		cornerHarris_demo(0, 0);
		//Canny implementation
		dst.create(src.size(), src.type()); //Create a matrix "dst" of the same type and size as src
		createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold); //Trackbar for Canny
		CannyThreshold(0, 0); // Show the image
		//Laplacian implementation
		createTrackbar("Sigma", window_Laplacian, &sigma, 15, 0); //create a named window for the trackbar and then the trackbar itself for Laplacian
		int ksize = (sigma * 5) | 1; 
		if (smoothType == GAUSSIAN) { GaussianBlur(original, smoothed, Size(ksize, ksize), sigma, sigma); } //if smoothType is GAUSSIAN
		else if (smoothType == BLUR) { blur(original, smoothed, Size(ksize, ksize)); } //if smoothType is BLUR
		else { medianBlur(original, smoothed, ksize); } //if smoothType is MEDIAN BLUR
		Laplacian(smoothed, laplace, CV_16S, 5); //Apply the Laplacian operator to the smoothed image and store it in "laplace"
		convertScaleAbs(laplace, result, (sigma + 1)*0.25); //For each array element, scales and calculates absolute values and converts them to a byte-sized number
		resize(result, finalLaplace, Size(0.32*result.cols, 0.32*result.rows), 0, 0, INTER_LINEAR); //resize the working image and store it in "src"
		char c = (char)waitKey(30); //int to char conversion
		if (c == ' ')
			smoothType = smoothType == GAUSSIAN ? BLUR : smoothType == BLUR ? MEDIAN : GAUSSIAN; //ternary operator; evaluates if GAUSSIAN is true, if so, returns BLUR, if not return BLUR, which can be MEDIAN or GAUSSIAN
		if (c == 'q' || c == 'Q' || c == 27) //if "q" has been pressed on the kayboard
			break;
		//Sobel & Scharr impelemtnation
		GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT); //apply a Gaussian mask to reduce noise
		//(input img, output img, output img depth, x derivative order, y derivative order...)
		Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT); Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		Scharr(src_gray, grad_x2, ddepth, 1, 0, scale, delta, BORDER_DEFAULT); Scharr(src_gray, grad_y2, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
		//scales and calculate absolute values for each array elements and otuputs the resultant array 
		convertScaleAbs(grad_x, abs_grad_x); convertScaleAbs(grad_y, abs_grad_y); convertScaleAbs(grad_x2, abs_grad_x2); convertScaleAbs(grad_y2, abs_grad_y2); //For Sobel, then for Scharr
		//Perform saturation arithmetic, meaning the range of ouput values is limited between a certain minimum and another maximum
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad); addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0, grad2);  //For Sobel, then for Scharr
		//Show resultant images
		imshow(window_Laplacian, finalLaplace); imshow(window_name, grad); imshow(window_name_2, grad2); imshow(corners_window, dst_norm_scaled);
	}
	waitKey(0); // Wait until user exit program by pressing a key
	return 0;
}
void CannyThreshold(int, void*) {
	blur(src_gray, detected_edges, Size(3, 3)); // Reduce noise with a blur of a 3x3 kernel
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size); //Apply Canny detector
	dst = Scalar::all(0); // initialize the destination image as a black canvas
	src.copyTo(dst, detected_edges); //copy "detected_edges" to "dst", which will be black everywhere but the points where the edges were detected
	imshow(window_name_3, dst); //Show result
}
void cornerHarris_demo(int, void*) {
	dst2 = Mat::zeros(src.size(), CV_32FC1); //initialize the destination image as a black canvas
	// Detecting corners
	cornerHarris(src_gray, dst2, blockSize, apertureSize, k, BORDER_DEFAULT); //Apply the harris corner detector technique (src array, dest, neighbourhood size, aperture param, free param, pixel interpolation method)
	// Normalizing
	normalize(dst2, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
	// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++) { //loop through the image height
		for (int i = 0; i < dst_norm.cols; i++) //loop through the image width
		{
			if ((int)dst_norm.at<float>(j, i) > thresh) //if the points at the dst_norm array are greater than the threshold...
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0); //...draw a circle with specified paramters around the corners
			}
		}
	}
}