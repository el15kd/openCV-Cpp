#include <iostream> //copies and pastes the header file into the code during preprocessing
#include <opencv2/highgui/highgui.hpp>
//a video is a stream of images over time
using namespace std; //groups all Cpp classes and functions under the name "std" so you don't need to type std::
using namespace cv; //container for all openCV classes and functions names
int main(int argc, const char**argv) {
	VideoCapture video(0); //function for video capturing; argument is the id of video capturing device, 0 is default device
	//store continuously captured images in "video"
	namedWindow("Video stream", CV_WINDOW_KEEPRATIO);
	double width = video.get(CV_CAP_PROP_FRAME_WIDTH); //get returns a specified property, width of frame
	double height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
	Size frameSize(static_cast<int>(width), static_cast<int>(height)); //convert double values into ints
	VideoWriter videowrite("C:/video.avi", CV_FOURCC('P', 'T', 'M', 'I'), 20, frameSize, true); //save video to disk 1 in MPEG-1 compression format
	//^^^ vidwrite is an object of VideoWriter; function object; VideoWriter is a func
	//and the arguments in () are property of VideoWriter
	//20 - frame rate; true sets colored mode
	while (true) { //infinite loop
		Mat frame;
		double temp = video.read(frame); //grabs, decodes and returns to variable stored in memory the next video frame
		videowrite.write(frame); //write frame to memory location
		imshow("Video Player", frame);
		waitKey(30); //ms
	}
	return 0;
}