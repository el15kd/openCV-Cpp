#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
int main(int argc, char**argv) {
	VideoCapture video_load(0);
	namedWindow("Adjust", CV_WINDOW_AUTOSIZE);
	int Hue_Lower_Value = 0, Hue_Upper_Value = 179;
	int Saturation_Lower_Value = 0, Saturation_Upper_Value = 255;
	int Value_Lower_Value = 0, Value_Upper_Value = 255;
	//Creates a user interface with a trackbar and attaches it to a window
	cvCreateTrackbar("Hue_Lower", "Adjust", &Hue_Lower_Value, 179), cvCreateTrackbar("Hue_Upper", "Adjust", &Hue_Upper_Value, 179);
	cvCreateTrackbar("Sat_Lower", "Adjust", &Saturation_Lower_Value, 255), cvCreateTrackbar("Sat_Upper", "Adjust", &Saturation_Upper_Value, 255);
	cvCreateTrackbar("Val_Lower", "Adjust", &Value_Lower_Value, 255); cvCreateTrackbar("Val_Upper", "Adjust", &Value_Upper_Value, 255);
	while (1) {
		Mat actual_Image; Mat convert_to_HSV; Mat detection_screen;
		bool temp = video_load.read(actual_Image); //grabs, decodes and returns the next video frame into the matrix "actual_Image"
		cvtColor(actual_Image, convert_to_HSV, COLOR_BGR2HSV); //change image colour space from RGB to HSV
		//inRange checks if the elements of one array lie between the elements of two other arrays; Scalar is a container for a maximum for values of double or smaller data type
		inRange(convert_to_HSV, Scalar(Hue_Lower_Value, Saturation_Lower_Value, Value_Lower_Value), Scalar(Hue_Upper_Value, Saturation_Upper_Value, Value_Upper_Value), detection_screen); //(src arr,lower boundary,upper,dest arr)
		//In image processing, Dilation means adding pixels to the boundaries of objects in an image and Erosion is the inverse of Dilation
		//These two morphological operations apply a structural element to an input to generate an output image
		//(src image, dest image, structural element)
		/* 
		//erode(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//dilate(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//dilate(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//erode(detection_screen, detection_screen, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		^^^may be used optionally
		*/
		imshow("Thresholded Image", detection_screen);
		imshow("Original", actual_Image);
		if (waitKey(30) == 27) { break; }
	}
		return 0;
	//^^^trackbar name, window name, pointer to an int whose value reflects the slider position
}