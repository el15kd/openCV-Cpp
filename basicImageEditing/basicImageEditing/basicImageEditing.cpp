#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //image processing library of openCV so as to be able to use the histogram equalizer and cvtColor
using namespace std;
using namespace cv;
int main(int argc, const char**argv) {
	//BsetBreakOnError(true);
	Mat original = imread("C:/Users/KonstantinDragostinov/Desktop/image", CV_LOAD_IMAGE_COLOR);
	Mat gray; //to be used for a grayscale image
	Mat brighter = original + Scalar(80, 80, 80);
	Mat darker = original - Scalar(80, 80, 80);
	Mat contrast; Mat hist; Mat hist2; Mat histeql;
	vector<Mat>channels; //vectors are like arrays except dynamycally sized
	original.convertTo(contrast, -1, 2, 0); //(new image matrix, keep depth (-1), constrast amount (times 2), scale factor, delta)
	cvtColor(original, gray, CV_BGR2GRAY); //convert original image to grayscale; (input arr, output arr, color space conversion)
	equalizeHist(gray, hist); //Apply histogram equaliser to a grayscale image; equalises the pixel values
	split(hist2,channels); //splits a multi-channel array into separate single-channel arrays
	equalizeHist(channels[2],channels[2]);
	merge(channels, hist);
	cvtColor();
	namedWindow("Original", CV_WINDOW_KEEPRATIO);
	namedWindow("Brighter", CV_WINDOW_KEEPRATIO);
	namedWindow("Darker", CV_WINDOW_KEEPRATIO);
	namedWindow("Contrast", CV_WINDOW_KEEPRATIO);
	namedWindow("Gray", CV_WINDOW_KEEPRATIO);
	namedWindow("Histogram", CV_WINDOW_KEEPRATIO);
	namedWindow("Histogram-2", CV_WINDOW_KEEPRATIO);
	imshow("Original", original);
	imshow("Brighter", brighter);
	imshow("Darker", darker);
	imshow("Contrast", contrast);
	imshow("Gray", gray);
	imshow("Histogram", hist);
	imshow("Histogram-2", hist);
	//An image histogram is a plot, for each pixel, of the pixel intensity (in x) corresponding to the number of pixels possessing this intensity (in y) (intensity vs pixel plot)
	//can use frequency (as a %) insteaad of number of pixels; how often a pixel intensity appears
	//can also use cumulative frequency, e.g. at some x-y location, 92.1% of pixels have an intensity value <= 189, 
	//^^^(non-decreasing function of intensity); ends with value of 1 since all pixels have intensity <= the maximum intensity
	//higher intensity, brighter, whiter color
	//increase contrast, make image sharper
	//Applying Histogram equalisation (a way of contrast adjustmnet) makes a frequency histogram spread out, more evenly distributed
	//allows for areas of lower contrast to gain a higher one
	waitKey(0);
	destroyAllWindows();
	return 0;
}
