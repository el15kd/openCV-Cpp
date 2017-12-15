#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
//1. Histogram equalizatino of RGB channels of original image
//2. Histogram equalizatino of grayscale of original image
//1. Histogram equalizatino of YCrCb channels of original image
//4. Histogram of RGB intensity distributions from 0 to 255
int main(int argc, const char*argv) {
	//Load the working image and declare dense array to store the images
	Mat gray, hist, hist2, histeql, origEq, Cb, r_hist, g_hist, b_hist;
	Mat original = imread("C:/Cat.jpg", CV_LOAD_IMAGE_UNCHANGED); //r/g/b_hist are used as objects to store the histograms
	// Here we will store the three YCrCb and RGB channels into three indices for both vectors
	vector<Mat>channels,channels2; //vectors are dynamically-sized sequence objects that provide array-style random access (using array[index]
	split(original, channels); // Separate the image in 3 places (B, G and R)
	//Histogram configureation 
	int histSize = 256, hist_w = 512, hist_h = 400, bin_w = cvRound((double)hist_w / histSize); // Establish the number of bins & histogram height and width
	float range[] = {0, 256}; const float* histRange = {range}; // Set the ranges for B,G,R
	bool uniform = true, accumulate = false; //need bins to have a uniform size and to clear the histogram in the beginning
	// Compute the histograms (pointers to the channels array, num of source arrays, channel to be measured (0 = intensity), mask to be used on the source array,...
	//...store the object in b/r/r_hist, 1 - histogram dimensionality, n of bins for each dimension, range of values to be measured for each dimension,...
	//...bin size are const & histogram is cleared at the beginning
	calcHist(&channels[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0)); // Create and image for and to draw the histograms for B, G and R
	// Normalize the result to [0,histImage.rows]; (in-, output array, lower and upper limits to normalize the value or r/g/b_hist,...
	//...type of normalization - between the aforementioned limits, output normalized array is same as input one, optional mask)
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// Draw for each channel
	for (int i = 1; i < histSize; i++) { //r/g/b_hist.at<float>(i) indicates the dimension (1D)
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	//More image manipulation
	cvtColor(original, gray, CV_BGR2GRAY); //convert original image to grayscale
	cvtColor(original, hist2, CV_BGR2YCrCb); //Convert RGB to YCrCb (Luma, Blue-difference, Red-difference)
	split(hist2, channels); //split the YCrCb image into its channels
	split(original, channels2); //split original RGB image into its channels
	//Histogram equalization adjusts contrast using the image's histogram
	equalizeHist(channels2[0], channels2[0]); merge(channels2, origEq);
	equalizeHist(channels2[0], channels2[0]); merge(channels2, origEq);
	equalizeHist(channels2[0], channels2[0]); merge(channels2, origEq);
	//Histogram equalization of grayscale image (therefore, it is applied to only 1 channel), then for RGB seperately
	equalizeHist(gray, hist);
	//Equalizing Y, then Cr, then Cb channels and merging them back to original image separately
	equalizeHist(channels[0], channels[0]); merge(channels, hist2); 
	equalizeHist(channels[1], channels[1]); merge(channels, hist2); 
	equalizeHist(channels[2], channels[2]); merge(channels, hist2);
	cvtColor(hist2, histeql, CV_YCrCb2BGR); //Convert back to RGB and store in histeql
	//original.rows and .cols get the image resolution; I want to make the images smaller to fit them on my screen
	int resX=round(original.cols*0.6); int resY=round(original.rows*0.6);
	//Name the windows, resize them and display the images
	namedWindow("Original image", CV_WINDOW_KEEPRATIO); resizeWindow("Original image", resX, resY);
	imshow("Original image", original); //Original
	namedWindow("Original Image Histogram", CV_WINDOW_KEEPRATIO); resizeWindow("Original Image Histogram", resX, resY);
	imshow("Original Image Histogram", histImage); //Original image histogram
	namedWindow("Original Image Equalized", CV_WINDOW_KEEPRATIO); resizeWindow("Original Image Equalized", resX, resY);
	imshow("Original Image Equalized", origEq); //Original image equalized RGB channels
	namedWindow("Original in Grayscale", CV_WINDOW_KEEPRATIO); resizeWindow("Original in Grayscale", resX, resY);
	imshow("Original in Grayscale", gray); //Grayscale 
	namedWindow("grayHistEqualization", CV_WINDOW_KEEPRATIO); resizeWindow("grayHistEqualization", resX, resY);
	imshow("grayHistEqualization", hist); //Equalized grayscale
	namedWindow("equalizeYCrCb", CV_WINDOW_KEEPRATIO); resizeWindow("equalizeYCrCb", resX, resY);
	imshow("equalizeYCrCb", histeql); //Original with equalized Y, Cr, Cb channels
	waitKey(0);
	return 0;
}