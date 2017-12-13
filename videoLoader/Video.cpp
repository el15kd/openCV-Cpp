#include <iostream> //copies and pastes the header file into the code during preprocessing
#include <opencv2/highgui/highgui.hpp>
//a video is a stream of images over time
using namespace std; //groups all Cpp classes and functions under the name "std" so you don't need to type std::
using namespace cv; //container for all openCV classes and functions 
int main(int argc, const char**argv) {
	VideoCapture video(0); //class for video capturing; argument is the id of video capturing device
	while (true) {
		Mat image;
		bool temp = video.read(image); //grabs, decodes and returns to variable the next video frame
		imshow("Video Player", image);
		waitKey(20);
	}
	return 0;
}