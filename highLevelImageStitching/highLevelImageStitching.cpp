#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
int main(int argc, char* argv[])
{
	Mat img1 = imread("C:/img1", CV_LOAD_IMAGE_UNCHANGED);
	Mat img2 = imread("C:/img2", CV_LOAD_IMAGE_UNCHANGED);
	vector<Mat> imgs = { img1 , img2 }; //Store the input images in a vector
	Stitcher::Mode mode = Stitcher::PANORAMA; //Define Stitcher function to panorama
	Mat pano; // Define object to store the stitched image
	Ptr<Stitcher> stitcher = Stitcher::create(mode, false); // Create a Stitcher class object with mode panoroma
	Stitcher::Status status = stitcher->stitch(imgs, pano); // Command to stitch all the images present in the image array
	if (status != Stitcher::OK) { // Check if images could not be stiched
		cout << "Can't stitch images\n"; // status is OK if images are stiched successfully
		return -1;
	}
	imwrite("Result.jpg", pano); // Store the resulting image stiched from the given set of images as "result.jpg"
	namedWindow("First image", WINDOW_NORMAL); imshow("First image", img1);
	namedWindow("Second image", WINDOW_NORMAL); imshow("Second image", img2);
	namedWindow("Result", WINDOW_NORMAL); imshow("Result", pano);
	waitKey(0);
	return 0;
}
