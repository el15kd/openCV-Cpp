#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //image processing library of openCV so as to be able to use the histogram equalizer and cvtColor
using namespace std;
using namespace cv; /*
void ShowManyImages(string title, int nArgs, ...) {
	int size, i, m, n, w, h; // w -  number of images in a column, h - number of images in a row, m - horizontal shift, n - vertical shift
	// If the number of arguments is lesser than 0 or greater than 12,  return without displaying
	if (nArgs <= 0) { printf("Number of arguments too small....\n"); return; }
	else if (nArgs > 14) {printf("Number of arguments too large, maimg.cols is 12\n"); return;}
	// Determine the size of the image, and the number of rows/cols from number of arguments
	else if (nArgs == 1) {w = h = 1;size = 300;} else if (nArgs == 2) {w = 2; h = 1;size = 300;}
	else if (nArgs == 3 || nArgs == 4) {w = 2; h = 2;size = 300;} else if (nArgs == 5 || nArgs == 6) {w = 3; h = 2;size = 200;} //2 ros and 3 columns
	else if (nArgs == 7 || nArgs == 8) {w = 4; h = 2;size = 200;} else {w = 4; h = 3;size = 150;}
	Mat temp, DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3); // Create a new 3 channel image
	va_list args; va_start(args, nArgs); // Used to get the arguments passed
	for (i = 0, m = 20, n = 0; 
		i < nArgs; 
		i++, m += (20 + size)) { // Loop for nArgs number of arguments
		Mat img = va_arg(args, Mat); // Get the Pointer to the IplImage
		int max = (img.cols > img.rows) ? img.cols : img.rows; // Find whether height or width is greater in order to resize the image
		float scale = (float)((float)max / size); // Find the scaling factor to resize the image
		if (i % w != 0 && m == 20) {m = 0;}
		if (i % w == 0 && m != 20) {m = 0; n += size;} // Used to Align the images
		Rect ROI(m, n, (int)(img.cols / scale), (int)(img.rows / scale)); // Set the image ROI (region interest) to display the current image; img.cols, y, width, height
		resize(img, temp, Size(ROI.width, ROI.height));
		temp.copyTo(DispImage(ROI)); // Resize the input image and copy the it to the Single Big Image
	} 
	namedWindow(title, 1); imshow(title, DispImage); // Create a new window, and show the Single Big Image
	waitKey();
	va_end(args); // End the number of arguments
}
*/
//void ShowManyImages(string title, int nArgs)
// Determine the size of the image and the number of rows/cols from number of arguments
void takeDFT(Mat& source, Mat& destination);
void recenterDFT(Mat& source);
void showDFT(Mat& source);
void invertDFT(Mat& source, Mat& destination);
int morph_elem = 0, morph_size = 0, morph_operator = 0;
int const max_operator = 4, max_elem = 2, max_kernel_size = 21; //variables used for Morphological operations, dilation and erosion
char* window_name = "Morphology Transformations Demo";
char* window_name_2 = "Original";
Mat src, dst;
Mat original = imread("E:/ferrariF1.jpg", CV_LOAD_IMAGE_COLOR), workingImg, contrast, gray, channels[3], hist;
int thresh = 100, max_thresh = 255;
RNG rng(12345);
void thresh_callback(int, void*);
void Morphology_Operations(int, void*);
int main(int argc, const char**argv) {
	resize(original, workingImg, Size(0.23*original.cols, 0.23*original.rows), INTER_LINEAR);
	Mat removeB = workingImg.clone(), removeG = workingImg.clone(), removeR = workingImg.clone(); //copies where R, G and B channels respectively will be removed
	Mat workingImgCopy = workingImg.clone(), workingImgCopy2 = workingImg.clone(), workingImgCopy3 = workingImg.clone(), workingImgCopy4 = workingImg.clone(), dftMat, dftReady, dftOfOriginal, invertedDft;
	for (int r = 0; r < workingImgCopy.rows; r++) { //loop trough the image height
		for (int c = 0; c < workingImgCopy.cols; c++) { //loop trough the image width
			removeB.at<Vec3b>(r, c)[0] = workingImgCopy.at<Vec3b>(r, c)[0] * 0; //remove B, since OpencV treats RGB as BGR	
		}
	}
	for (int r = 0; r < workingImgCopy2.rows; r++) { //loop trough the image height
		for (int c = 0; c < workingImgCopy2.cols; c++) { //loop trough the image width
			removeG.at<Vec3b>(r, c)[1] = workingImgCopy2.at<Vec3b>(r, c)[1] * 0; //remove G, since OpencV treats RGB as BGR
		}
	}
	for (int r = 0; r < workingImgCopy3.rows; r++) { //loop trough the image height
		for (int c = 0; c < workingImgCopy3.cols; c++) { //loop trough the image width
			removeR.at<Vec3b>(r, c)[2] = workingImgCopy3.at<Vec3b>(r, c)[2] * 0; //remove R, since OpencV treats RGB as BGR
		}
	}
	//Morphological operations
	namedWindow( window_name, CV_WINDOW_AUTOSIZE);
	createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations);  // Create Trackbar to select Morphology operation
	createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,&morph_elem, max_elem,Morphology_Operations); // Create Trackbar to select kernel type
	createTrackbar("Kernel size: 2n +1", window_name,&morph_size, max_kernel_size,Morphology_Operations); // Create Trackbar to choose kernel size																	 /// Default start
	Morphology_Operations(0, 0); // Default start
	cvtColor(workingImgCopy4, gray, CV_BGR2GRAY); 
	//Finding & Drawing contours
	namedWindow(window_name_2, CV_WINDOW_AUTOSIZE); imshow(window_name_2, workingImg);
	createTrackbar(" Canny thresh:", window_name_2, &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);
	//Histogram equalization
	equalizeHist(gray, hist);
	//DFT
	Mat workingImgCopy5 = gray.clone();
	workingImgCopy5.convertTo(dftMat, CV_32FC1, 1.0 / 255.0); //Converts the array to a 32-bit depth one with 1 channel and normalises the values
	takeDFT(dftMat, dftOfOriginal); showDFT(dftOfOriginal); invertDFT(dftOfOriginal, invertedDft);
	Mat brigther = workingImg + Scalar(80, 80, 80), darker = workingImg - Scalar(80, 80, 80);
	workingImg.convertTo(contrast, -1, 1.5, 0);
	//ShowManyImages("Image", 6, img1, img2, img3, img4, img5, img6);
	//For exercise, I will try to resize the original image further using the "at" method of accessing pixels
	imshow("Original", workingImg); imshow("Brighter", brigther); imshow("Darker", darker);
	imshow("Contrast", contrast); imshow("Applying Histogram Equalization to a Grayscale image", hist);
	imshow("Blue, removed", removeB); imshow("Green, removed", removeG); imshow("Red, removed", removeR); imshow("DFT inverted = Grayscale of Original", invertedDft);
	//imshow("Red + Green", removeB); imshow("Red + Blue", removeG); imshow("Green + Blue", removeR); 
		//An image histogram is a plot, for each piimg.colsel, of the piimg.colsel intensity (in img.cols) corresponding to the number of piimg.colsels possessing this intensity (in y) (intensity vs piimg.colsel plot)
		//can use frequency (as a %) insteaad of number of piimg.colsels; how often a piimg.colsel intensity appears
		//can also use cumulative frequency, e.g. at some img.cols-y location, 92.1% of piimg.colsels have an intensity value <= 189, 
		//^^^(non-decreasing function of intensity); ends with value of 1 since all piimg.colsels have intensity <= the maimg.colsimum intensity
		//higher intensity, brighter, whiter color
		//increase contrast, make image sharper
		//Applying Histogram equalisation (a way of contrast adjustmnet) makes a frequency histogram spread out, more evenly distributed
		//allows for areas of lower contrast to gain a higher one
	waitKey(0); return 0;
}
void takeDFT(Mat& source, Mat& destination) {
	Mat dftMat, dftReady, dftOfOriginal;
	Mat dftComplex[2] = { source, Mat::zeros(source.size(), CV_32F) };//need a mat object that can hold real and imaginary components
	merge(dftComplex, 2, dftReady);
	dft(dftReady, dftOfOriginal, DFT_COMPLEX_OUTPUT); //Performs Direct Fourier Transform on an image
	destination = dftOfOriginal;
}
void recenterDFT(Mat& source) {//Recenter the DFT so that the low frequency information is in the center, and the high frequency one is the corners
	int centerX = source.cols / 2;//4 Mat objects for each quadrant, and switch the places of the quadrants such that (m11 = 4, m12 = 3, m21 = 2, m22 = 1)
	int centerY = source.rows / 2;
	//Get the quadrants; Rect(origin, size)
	Mat q1(source, Rect(0, 0, centerX, centerY)); Mat q2(source, Rect(centerX, 0, centerX, centerY));
	Mat q3(source, Rect(0, centerY, centerX, centerY)); Mat q4(source, Rect(centerX, centerY, centerX, centerY));
	Mat swapMap;//temporarily variable to hold info while swapping MATs
	q1.copyTo(swapMap); q4.copyTo(q1); swapMap.copyTo(q4);
	q2.copyTo(swapMap); q3.copyTo(q2); swapMap.copyTo(q3);
}
void showDFT(Mat& source) { //two channels coming in, corresponding to real and imaginary components
	Mat splitArray[2] = { Mat::zeros(source.size(),CV_32F),Mat::zeros(source.size(),CV_32F) };
	split(source, splitArray);//split the two channels
	Mat dftMagnitude;//Get magnitude using Pythagoras
	magnitude(splitArray[0], splitArray[1], dftMagnitude);
	dftMagnitude += Scalar::all(1); //add 1 to every single element of the array
	log(dftMagnitude, dftMagnitude);
	normalize(dftMagnitude, dftMagnitude, 0, 1, CV_MINMAX);
	recenterDFT(dftMagnitude); //passed by reference
	imshow("DFT", dftMagnitude);
}
void invertDFT(Mat& source, Mat& destination) {//convert frequency domain back to spatial domain
	Mat inverse;
	dft(source,inverse,DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE); //use bitwise OR to chain flags; rescale to a range between 0 and 1
	destination = inverse;
}
void Morphology_Operations(int, void*) {
	int operation = morph_operator + 2; // Since MORPH_X : 2,3,4,5 and 6
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(original, dst, operation, element); // Apply the specified morphology operation
	resize(dst, dst, Size(0.23*original.cols, 0.23*original.rows), INTER_LINEAR);
	imshow(window_name, dst);
}
void thresh_callback(int, void*) {
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Canny(gray, canny_output, thresh, thresh * 2, 3); // Detect edges using canny
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0)); // Find contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3); 
	for (int i = 0; i< contours.size(); i++) { // Draw contours
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}
