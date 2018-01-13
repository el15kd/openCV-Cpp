#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp" 
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
Mat img1, img2, templ, result, img1Copy, img2Copy, img_matches; //n-dimensional dense array image containers used for all the techniques explored bloew
//Used for Template Matching; Template matching finds areas in an image similar to a template image or patch
//Template is compared by sliding (move it one pixel at a time, left to right, up to down) it over the source image
//At each location, a metric is calculated defining how similar the patch is to the area it is compared to
const char* Template = "Template, Enlarged";
int match_method, max_Trackbar = 5;
void MatchingMethod(int, void*);
//Used for Perspective Correction by Homography; Homography is a 3x3 matrix transformation mapping a (set of) point(s) from one image to another (set of) point(s) in another image
//For two different image planes, two homographies are needed; need 4 corresponding pts - these are found by feature matching by, e.g., SIFT Or SURF
struct userdata {
	Mat im;
	std::vector<Point2f> points;
};
void mouseHandler(int event, int x, int y, int flags, void* data_ptr);
int main(int argc, char** argv) {

	//Input the working images and resize them appropriately

	img1 = imread("E:/book1.jpg"), templ = imread("E:/book1-patch.jpg"), img2 = imread("E:/book2.jpg"); // Load image and template
	resize(img1, img1, Size(0.13*img1.cols, 0.13*img1.rows), 1, 1, INTER_LINEAR);
	resize(templ, templ, Size(1.7*templ.cols, 1.7*templ.rows), 1, 1, INTER_LINEAR);
	resize(img2, img2, Size(0.13*img2.cols, 0.13*img2.rows), 1, 1, INTER_LINEAR);
	img1Copy = img1.clone(); img2Copy = img2.clone();
	//TEMPLATE MATCHING
	namedWindow("Source Image", WINDOW_AUTOSIZE); namedWindow("Result window", WINDOW_NORMAL);
	createTrackbar("Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED", "Source Image", &match_method, max_Trackbar, MatchingMethod); // Create Trackbar; used for selection of Template Matching method
	MatchingMethod(0, 0);

	//PERSPECTIVE CORRECTION BY HOMOGRAPHY

	Size size(300, 400); Mat im_dst = Mat::zeros(size, CV_8UC3); // Destination image and aspect ratio of book (3:4)
	std::vector<Point2f> pts_dst; // Create a vector of destination points.
	pts_dst.push_back(Point2f(0, 0));
	pts_dst.push_back(Point2f(size.width - 1, 0));
	pts_dst.push_back(Point2f(size.width - 1, size.height - 1));
	pts_dst.push_back(Point2f(0, size.height - 1));
	// Set data for mouse event
	userdata data;
	data.im = img2Copy;
	std::cout << "Click on the four corners of the book -- top left first and\n";
	std::cout << "bottom left last -- and then hit ENTER\n";
	// Show image and wait for 4 clicks. 
	imshow("Destionation Image", img2Copy);
	setMouseCallback("Destionation Image", mouseHandler, &data); // Set the callback function for any mouse event
	waitKey(0);
	Mat h = findHomography(data.points, pts_dst); // Calculate the homography
	warpPerspective(img2, im_dst, h, size); //apply the homography
	imshow("Perspective corrected", im_dst);

	//HOMOGRAPHY by MANUALLY SELECTING THE CORRESPONDING PTS

	// Four corners of the book in source image
	std::vector<Point2f> pts_src; std::vector<Point2f> pts_dst2;
	pts_src.push_back(Point2f(0.2 * 1113, 0.2 * 1093)); pts_src.push_back(Point2f(0.2 * 1601, 0.2 * 1541));
	pts_src.push_back(Point2f(0.2 * 1361, 0.2 * 797)); pts_src.push_back(Point2f(0.2 * 921, 0.2 * 2261));
	// Four corners of the book in destination image.
	pts_dst2.push_back(Point2f(0.2 * 1742, 0.2 * 445)); pts_dst2.push_back(Point2f(0.2 * 1526, 0.2 * 959));
	pts_dst2.push_back(Point2f(0.2 * 2096, 0.2 * 483)); pts_dst2.push_back(Point2f(0.2 * 767, 0.2 * 792));

	// Calculate Homography
	Mat h2 = findHomography(pts_src, pts_dst2);

	// Output image
	Mat im_out;
	// Warp source image to destination based on homography
	warpPerspective(img1Copy, im_out, h, img2Copy.size());
	// Display images
	imshow("Source Image", img1Copy); imshow("Warped Source Image", im_out);
	
	//HOmoGRAPHY by SURF
	
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	detector->detect(img1, keypoints_object);
	detector->detect(img2, keypoints_scene);
	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<SURF> extractor = SURF::create();
	Mat descriptors_object, descriptors_scene;
	extractor->compute(img1, keypoints_object, descriptors_object);
	extractor->compute(img2, keypoints_scene, descriptors_scene);
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);
	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptors_object.rows; i++) { //-- Quick calculation of max and min distances between keypoints
		double dist = matches[i].distance;
		if (dist < min_dist) { min_dist = dist; }
		if (dist > max_dist) {max_dist = dist;}
	}
	printf("-- Max dist : %f \n", max_dist); printf("-- Min dist : %f \n", min_dist);
	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_object.rows; i++) {
		if (matches[i].distance < 3 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}
	drawMatches(img1, keypoints_object, img2, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Localize the object
	std::vector<Point2f> obj; std::vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++) {
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, RANSAC);
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
	obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(img1.cols, 0), scene_corners[1] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(img1.cols, 0), scene_corners[2] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(img1.cols, 0), scene_corners[3] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(img1.cols, 0), scene_corners[0] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);
	waitKey(0);
	return 0;
}
void MatchingMethod(int, void*) { //used as a callback for createTrackbar
	Mat img_display; // Source image to display
	img1.copyTo(img_display);
	int result_cols = img1.cols - templ.cols + 1, result_rows = img1.rows - templ.rows + 1; // Create the result matrix
	result.create(result_cols, result_rows, CV_32FC1);
	// Do the Matching and Normalize
	matchTemplate(img1, templ, result, match_method); normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	// Localizing the best match with minMaxLoc
	double minVal, maxVal; 
	Point minLoc, maxLoc,matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) { matchLoc = minLoc; } // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	else { matchLoc = maxLoc; }
	//Draw a rectangel around the area corresponding to the highest match and display results
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	imshow("Source Image", img_display); imshow("Result window", result); imshow(Template, templ);
	return;
}
void mouseHandler(int event, int x, int y, int flags, void* data_ptr) {
	if (event == EVENT_LBUTTONDOWN) {
		userdata *data = ((userdata *)data_ptr);
		circle(data->im, Point(x, y), 3, Scalar(0, 0, 255), 5, CV_AA);
		imshow("Destionation Image", data->im);
		if (data->points.size() < 4) {
			data->points.push_back(Point2f(x, y));
		}
	}
}