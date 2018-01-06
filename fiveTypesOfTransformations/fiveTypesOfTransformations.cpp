#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//Explored are: Scaling; Rotation; Reflection; Affine transf; Perspective Transf;
//1. An image pyramid is a collection of layered images arising from a single original one that are successively downsampled until a certain resolution
//The higher the layer, the smaller the size. Size reduction means information loss. To up/down sample an image, convolve with a Gaussian kernel.
//2. Remapping is the process of taking pixels from one place and transferring them elsewhere in the image. Here we explore, reflections in X, in Y, and in XY
//as well as flipping and rotating
//3. An Affine transformation between two images involving translation + rotation + scale + warp + shear
//4. Homography is almost the same as a Perspective Transform
using namespace cv;
using namespace std;
Mat src = imread("E:/owl.jpg"), src2 = imread("E:/owl2.jpg"), dst, tmp, warp_dst, warp_rotate_dst;
Mat upsideDownX, upsideDownY, upsideDownDest;
Mat xxReflect, xyReflect, xReflectDest;
Mat yxReflect, yyReflect, yReflectDest;
Mat xyReflectX, xyReflectY, xyReflectDest;
Mat cwRotDest, ccwRotDest, perspectiveTOutput;
void upDownScale(), remap(), affine(), perspectiveTr();
int ind = 0;
void flip(), reflectX(), reflectY(), reflectXY(), rotate90CW(), rotate90CCW();
const char* pyramid_window = "Pyramids Demo"; const char* source_window = "Source image";
const char* warp_window = "Warp"; const char* warp_rotate_window = "Warp + Rotate";
Point2f srcTri[3], dstTri[3];
Mat rot_mat(2, 3, CV_32FC1), warp_mat(2, 3, CV_32FC1);
int main(int argc, char** argv) {
	//Scale down 5 times the input img (from (960x540) to 192x108)) so I can fit 10 of them on my 1920x1080 screen
	resize(src, src, Size(src.cols / 2, src.rows / 2), 1, 1, INTER_LINEAR);
	remap(); affine(); perspectiveTr(); upDownScale();
	waitKey(0); return 0;
}
void upDownScale() {
	printf("\n Zoom In-Out demo  \n "); printf("------------------ \n");
	printf(" * [u] -> Zoom in  \n"); printf(" * [d] -> Zoom out \n");
	printf(" * [ESC] -> Close program \n \n");
	if (!src.data) {printf(" No data! -- Exiting the program \n");}
	tmp = src; dst = tmp;
	namedWindow(pyramid_window, WINDOW_AUTOSIZE); imshow(pyramid_window, dst);
	while (true) {
		int c = waitKey(10);
		if ((char)c == 27) {break;}
		if ((char)c == 'u') {
			pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
			printf("** Zoom In: Image x 2 \n");
		}
		else if ((char)c == 'd') {
			pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
			printf("** Zoom Out: Image / 2 \n");
		}
		imshow(pyramid_window, dst);
		tmp = dst;
	}
}
void remap() {
	upsideDownX.create(src.size(), CV_32FC1); upsideDownY.create(src.size(), CV_32FC1);
	xxReflect.create(src.size(), CV_32FC1); xyReflect.create(src.size(), CV_32FC1);
	yxReflect.create(src.size(), CV_32FC1); yyReflect.create(src.size(), CV_32FC1);
	xyReflectX.create(src.size(), CV_32FC1); xyReflectY.create(src.size(), CV_32FC1);
	rotate90CW(); rotate90CCW();
	//namedWindow("CW Rot",WINDOW_NORMAL); namedWindow("CCW Rot", WINDOW_NORMAL);
	//resizeWindow("CW Rot", 480, 270); resizeWindow("CCW Rot", 480, 270);
	flip(); remap(src, upsideDownDest, upsideDownX, upsideDownY, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	reflectX(); remap(src, xReflectDest, xxReflect, xyReflect, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	reflectY(); remap(src, yReflectDest, yxReflect, yyReflect, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	reflectXY(); remap(src, xyReflectDest, xyReflectX, xyReflectY, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	imshow("CW Rot", cwRotDest); imshow("CCW Rot", ccwRotDest);
	imshow("Original", src); imshow("Flip", upsideDownDest);
	imshow("Reflect-X", xReflectDest); imshow("Reflect-Y", xReflectDest); imshow("Reflect-XY", xyReflectDest);
}
void flip() {
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			upsideDownX.at<float>(j, i) = i;
			upsideDownY.at<float>(j, i) = src.rows - j;
		}
	}
}
void reflectX() {
for (int j = 0; j < src.rows; j++) {
	for (int i = 0; i < src.cols; i++) {
		xxReflect.at<float>(j, i) = src.cols - i;
		xyReflect.at<float>(j, i) = j;
		}
	}
}
void reflectY() {
for (int j = 0; j < src.rows; j++) {
	for (int i = 0; i < src.cols; i++) {
		yxReflect.at<float>(j, i) = i;
		yyReflect.at<float>(j, i) = src.rows - j;
		}
	}
}
void reflectXY() {
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			xyReflectX.at<float>(j, i) = src.cols - i;
			xyReflectY.at<float>(j, i) = src.rows - j;
		}
	}
}
Point2f cxcy(src2.cols / 2.0F, src2.rows / 2.0F); //get image center point
void rotate90CW() {
	Mat cwRot = getRotationMatrix2D(cxcy,-90,1.0); warpAffine(src, cwRotDest, cwRot, src2.size());
}
void rotate90CCW() {
	Mat ccwRot = getRotationMatrix2D(cxcy,90,1.0); warpAffine(src, ccwRotDest, ccwRot, src2.size());
}
void affine() {
	warp_dst = Mat::zeros(src.rows, src.cols, src.type()); // Set the dst image the same type and size as src
	// Set your 3 points to calculate the  Affine Transform
	srcTri[0] = Point2f(0, 0); dstTri[0] = Point2f(src.cols*0.0, src.rows*0.33);
	srcTri[1] = Point2f(src.cols - 1, 0); dstTri[1] = Point2f(src.cols*0.85, src.rows*0.25);
	srcTri[2] = Point2f(0, src.rows - 1); dstTri[2] = Point2f(src.cols*0.15, src.rows*0.7);
	// Get the Affine Transform & Apply it to the src img
	warp_mat = getAffineTransform(srcTri, dstTri); warpAffine(src, warp_dst, warp_mat, warp_dst.size());
	//Rotating the image after Warp: Compute a rotation matrix with respect to the center of the image
	Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
	double angle = -50.0, scale = 0.6;
	// Get the rotation matrix with the specifications above
	rot_mat = getRotationMatrix2D(center, angle, scale);
	// Rotate the warped image
	warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());
	imshow(warp_window, warp_dst); imshow(warp_rotate_window, warp_rotate_dst);
}
void perspectiveTr() {
	// Input Quadilateral or Image plane coordinates & Output Quadilateral or World plane coordinates respectively
	Point2f inputQuad[4], outputQuad[4];
	Mat lambda(2, 4, CV_32FC1); //The perspective transform matrix
	lambda = Mat::zeros(src.rows, src.cols, src.type());
	// The 4 points that select quadilateral on the input , from top-left in clockwise order; These four pts are the sides of the rect box used as input 
	inputQuad[0] = Point2f(-30, -60); inputQuad[1] = Point2f(src.cols + 50, -50);
	inputQuad[2] = Point2f(src.cols + 100, src.rows + 50); inputQuad[3] = Point2f(-50, src.rows + 50);
	// The 4 points where the mapping is to be done , from top-left in clockwise order
	outputQuad[0] = Point2f(0, 0); outputQuad[1] = Point2f(src.cols - 1, 0);
	outputQuad[2] = Point2f(src.cols - 1, src.rows - 1); outputQuad[3] = Point2f(0, src.rows - 1);
	lambda = getPerspectiveTransform(inputQuad, outputQuad); //get the perspective transform matrix
	warpPerspective(src, perspectiveTOutput, lambda, perspectiveTOutput.size());
	imshow("Perspective Transform", perspectiveTOutput);
}