#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
int thresh = 100, max_thresh = 255;
RNG rng(12345);
void thresh_callback(int, void*);
Mat dst, cdst, gray, src_gray;
int main(int argc, char** argv) {
	Mat src = imread("E:/cookies.png", 1);
	
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	Mat standardHough = cdst.clone(), probabilisticHough = cdst.clone(), circ = src.clone();
	//Standard Hough line transform
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI / 45, 100, 0, 0);
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta), x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b)); pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b)); pt2.y = cvRound(y0 - 1000 * (a));
		line(standardHough, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
	//Probabilistic Hough line transforms
	vector<Vec4i> lines2;
	HoughLinesP(dst, lines2, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines2.size(); i++) {
		Vec4i l = lines2[i];
		line(probabilisticHough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
	imshow("Standard Hough Transform", standardHough);
	imshow("Probabilistic Hough Transform", probabilisticHough);
	//Hough Circle Transform
	vector<Vec3f> circles;
	GaussianBlur(circ, circ, Size(9, 9), 2, 2);//Reduce noise to avoid false circle detection by blurring
	cvtColor(circ, gray, COLOR_BGR2GRAY); //convert input image to grayscale
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 150, 30, 0, 0);//Apply Hough Transform for cicles
																				 //(input image, (x,y,r) values of detected circles, detection methods, inverse ratio of resolution...
																				 //...minimum distance between detected centers, upper threshold for canny edge detector, threshold for center detection
	for (size_t i = 0; i < circles.size(); i++) { //draw the detected circles
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(circ, center, 3, Scalar(0, 255, 0), -1, 8, 0); // circle center
		circle(circ, center, radius, Scalar(0, 0, 255), 3, 8, 0); // circle outline
	}
	imshow("Hough Circle Transform", circ);
	//Contour detection by Canny Edge detector
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	const char* source_window = "Source";
	namedWindow(source_window, WINDOW_AUTOSIZE);
	createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);
	imshow(source_window, src);
	waitKey(0); return 0;
}
void thresh_callback(int, void*)
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	// Find contours
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}
	// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	Mat drawing2 = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++) {
		mu[i] = moments(contours[i], false); //get the moments
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00); //get the mass centers
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength(contours[i], true));
		
		drawContours(drawing2, contours, i, color, 2, 8, hierarchy, 0, Point());
		rectangle(drawing2, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		circle(drawing2, center[i], (int)radius[i], color, 2, 8, 0);
		circle(drawing2, mc[i], 4, color, -1, 8, 0);
	}
	// Show in a window
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing); imshow("Shapes over the contours", drawing2);
	/// Calculate the area with the moments 00 and compare with the result of the OpenCV function
	printf("\t Info: Area and Contour Length \n");
}