#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp> //for the random number generatroz
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
//Posterization/Quantization (can be used interchangibly in this example) reduce the number of colors in an image
int main(int /*argc*/, char** /*argv*/) {
	const int MAX_CLUSTERS = 5; //The "K" in k-means
	Scalar colorTab[] = {Scalar(0, 0, 255), Scalar(0,255,0), Scalar(255,100,100), Scalar(255,0,255), Scalar(0,255,255)}; //Array of type Scalar storing colors for the circles to be drawn; 
	Mat centers, labels, img = imread("E:/leeds.jpg"), img2(img.size(), img.type(),img.type()), img3(img.size(),img.type()); //same the as resolution and the type of the input image
	img2 = Scalar(255); //Make img2 a white canvas by filling it with white 
	Size size = img.size(); //get the image dimensions...
	int attempts = 8; //Attempts for K-means
	int resizedHeight = int(0.5*size.height), resizedWidth = int(0.5*size.width); //...and resize the final images for "imshow"
	RNG rng(12345); //rng object of RNG class
	int i, k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1), sampleCount = rng.uniform(1, 1001);
	Mat points(sampleCount, 1, CV_32FC2);
	clusterCount = MIN(clusterCount, sampleCount); //return the smaller of the two numbers
	img2 = Scalar::all(255); //White image where the k-means dots will be put
	for (;;) { 
		//1. Quantization 
		Mat samples(img.rows * img.cols, 3, CV_32F);
		for (int y = 0; y < img.rows; y++) //loop through the height of the image...
			for (int x = 0; x < img.cols; x++) //...and through the width...
				for (int z = 0; z < 3; z++) //...and through the channels
					samples.at<float>(y + x*img.rows, z) = img.at<Vec3b>(y, x)[z];
		kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
		for (int y = 0; y < img.rows; y++) //loop through the height of the image...
			for (int x = 0; x < img.cols; x++) { //...and through the width
				int cluster_idx = labels.at<int>(y + x*img.rows, 0); //index the labels array elements and convert them to int after doing a calculation
				img3.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0); //render the image
				img3.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
				img3.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
			}
		//2. Finding k-means clusters and drawing them seperately and over the original image
		for (k = 0; k < clusterCount; k++) { /* generate random sample from multigaussian distribution */
			Point center; //tuple of x and y coordinates
			center.x = rng.uniform(0, img.cols); center.y = rng.uniform(0, img.rows);
			circle(img, center, 5, Scalar(0,0,0), FILLED, LINE_AA); //Draw center of cluster with a black dot
			circle(img2, center, 5, Scalar(0,0,0), FILLED, LINE_AA);
			Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
				k == clusterCount - 1 ? sampleCount :
				(k + 1)*sampleCount / clusterCount);
			rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
		}
		randShuffle(points, 1, &rng); //shuffles array elements randomly
		kmeans(points, clusterCount, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			attempts, KMEANS_PP_CENTERS, centers);
		for (i = 0; i < sampleCount; i++) {
			int clusterIdx = labels.at<int>(i); //get the i-th element of the labels array and store it in clusterIdx
			Point ipt = points.at<Point2f>(i); //Get circle center
			circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA); //draw the circles corresponding to the points
			circle(img2, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA); //draw the circles corresponding to the points
		}

		//3. Show outputs
		resize(img, img, Size(resizedWidth, resizedHeight)); resize(img2, img2, Size(resizedWidth, resizedHeight)); resize(img3, img3, Size(resizedWidth, resizedHeight));
		imshow("Clusters in-image", img); imshow("Clusters alone", img2); imshow("clustered image, Attemps = 8", img3);
		waitKey(0);
	}
	return 0;
}