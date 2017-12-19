#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
int main(int argc, char* argv[]) {
	//Observing the effect of various image filters using equal-sized (5x5) kernels
	Mat original = imread("E:/noise.jpg"), img, img2, img3, img4, img5, img6; //used for storing source and resultnat images
	resize(original, img, Size(0.3*original.cols, 0.3*original.rows), 0, 0, INTER_LINEAR); //resize the input image
	medianBlur(img, img2, 5); GaussianBlur(img, img3, Size(5, 5),0,0, BORDER_CONSTANT); //median and Gaussian blur
	bilateralFilter(img, img4, 15, 80, 80, BORDER_DEFAULT); boxFilter(img, img5, -1, Size(5,5)); //bilateral and vox filters
	blur(img, img6, Size(5,5),Point(-1,-1),BORDER_DEFAULT); //normalised box filter
	imshow("Original, resized", img); imshow("medianBlur", img2);  imshow("gaussianBlur", img3);
	imshow("bilateralFilter", img4); imshow("boxFilter", img5); imshow("normalizedBoxFilter", img6);
	waitKey(0);
	return 0;
}