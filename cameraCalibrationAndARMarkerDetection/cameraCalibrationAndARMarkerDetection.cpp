#include "opencv2\core.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\aruco.hpp"
#include "opencv2\calib3d.hpp"
#include <sstream>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;
const float calibrationSquareDimension = 0.027f; //in meters
const float arucoSquareDimension = 0.144f; //in meters
const Size chessboardDimensions = Size(6, 9);
//9x6 refers to the intersections/borders between black and white; each square is 27x27 mm
void createArucoMarker() { //detect pose of the markers with a camera
	Mat outputMarker; //object for aruco marker img
	//a pointer to the aruco marker dictionary obj; will continualy move through all these markers and print out each one to our directory => get a series of potentially usable imgs
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50); //smaller marker => faster detection
	for (int i = 0; i < 50; i++) { //iterate over the 50 markers and write them to a directory
		aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);//draw the marker itself; which marker from the dictionary we want to push into the Mat obj
		ostringstream convert; //take the string name, append to it the i-th value and use it as the file name when we write to the directory
		string imageName = "4x4Marker_"; //E.g. 4x4Marker_1; 4x4Marker_2, etc.
		convert << imageName << i << ".jpg";
		imwrite(convert.str(),outputMarker); //convert ostringstream to a string to be used by imwrite
	}
}
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners) { //vector will all the pts we calculate
	for (int i = 0; i < boardSize.height;i ++) {
		for (int j = 0; j < boardSize.width;j++) { //z = 0.0f because it's a flat plane
			corners.push_back(Point3f(j*squareEdgeLength,i*squareEdgeLength,0.0f)); //push into the corners vector of the calculated locations where we expect these things to be
		}
	}
}
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false) { //extract detected chessboard corners; need to take in all the diff imgs we are working with
	//^^Find & potentially visualise any chessboard corners; reference "allFoundCorners" (intersection of chessboard locations) to vector inside a vector 
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++) { //use iterators
		vector<Point2f> pointBuf;//need a buffer to hold all the pts found in the img if detected; Point2F because objects are 2D
		bool found = findChessboardCorners(*iter,Size(9,6),pointBuf,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		if (found) { //push the info onto our found corners; pattern found
			allFoundCorners.push_back(pointBuf);
		}
		if (showResults) {
			drawChessboardCorners(*iter, Size(9,6),pointBuf,found); //take in the corners and draw them onto the img
			imshow("Looking for Corners", *iter);
			waitKey(0);
		}
	}
}
//camera calibration function will take in images that have been validated as providing good info for calibration, i.e. chessboard can be found
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients) {
	vector<vector<Point2f>>checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false); //dont want to see output
	vector<vector<Point3f>> worldSpaceCornerPoints(1);//world space coord
	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);
	vector<Mat> rVectors, tVectors; //radial & tangential vectors
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);
	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints,boardSize,cameraMatrix,distanceCoefficients,rVectors,tVectors);
}
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients) {
	ofstream outStream(name);
	if (outStream) { //push out general info, such as rows and columns
		uint16_t rows = cameraMatrix.rows; uint16_t columns = cameraMatrix.cols;
		outStream << rows << endl; outStream << columns << endl; //push out num of rows & columns 
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				double value = cameraMatrix.at<double>(r,c); //take value out of camera matrix (64-bit) & store in the temporary var
				outStream << value << endl;
			}
		}
		rows = distanceCoefficients.rows; columns = distanceCoefficients.cols;
		outStream << rows << endl; outStream << columns << endl;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				double value = distanceCoefficients.at<double>(r, c); //take value out of camera matrix (64-bit) & store in the temporary var
				outStream << value << endl;
			}
		}
		outStream.close(); return true;
	}
	return false;
}
//now that the camera has been calibrated, load the data back in & use it to populate the aruco marker info & allow the algorithm to find the markers and move around
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients) {
	ifstream inStream(name);//input file stream
	if (inStream) { //if stream found
		uint16_t rows; uint16_t columns;
		inStream >> rows; inStream >> columns; //push info into rows & columns
		cameraMatrix = Mat(Size(columns, rows), CV_64F); //create camera matrix; 64-bit float = double
		for(int r=0;r<rows;r++) { //iterate over & grab all camera matrix values
			for (int c = 0; c < columns; c++) { //iterate over & grab all distance coefficients
				double read = 0.0f;
				inStream >> read; //push that info read
				cameraMatrix.at<double>(r, c) = read;//push the info in
				cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}
		inStream >> rows; inStream >> columns;//grab distance coefficients
		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);
		for (int r = 0; r<rows; r++) { 
			for (int c = 0; c < columns; c++) {
				double read = 0.0f; inStream >> read;
				distanceCoefficients.at<double>(r, c) = read;
				cout << distanceCoefficients.at<double>(r, c) << "\n";
			}
		}
		inStream.close(); return true;
	}
	return false; //else
}
int startWebcamMonitoring(const Mat& cameraMatrix,const Mat& distanceCoefficients, float arucoSquareDimensions) { //return an int value denoting whether webcam monitoring has started; Met& are const references
	Mat frame; vector<int> markerIds; //need to know what the marker IDs are; vectors of those objs, a series of ints
	vector<vector<Point2f>> markerCorners, rejectedCandidates; //vector of vectors of pts
	aruco::DetectorParameters parameters; //parameters to be used for detection
	//make sure we're working with the same dictionary of aruco makers; Go for the smallers dictionary size & smallest number of markers since the more markers you add, the more work the pc has to do whether it's the right one
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);//since we could using different dictionaries, we need to pick the right one
	VideoCapture vid(0);
	if (!vid.isOpened()) { return -1; }
	namedWindow("Webcam",CV_WINDOW_AUTOSIZE); //Gui window
	vector<Vec3d> rotationVectors, translationVectors;
	while(true) { //loop trying to find the aruco markers
		if (!vid.read(frame)) break; //if we can't read a frame from the video, break, so the program doesn't try to read it forever
		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds); //this function will populate markerCorners & markerIds; markers ids are 0-49
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);
		for (int i = 0; i < markerIds.size();i++) { //iterate over all markers; if one is found, draw an axis ontop of it
			aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f); //draw the axes of the detected aruco marker ontop of the image frame
		}
		imshow("Webcam", frame); 
		if (waitKey(30) >= 0) break; //if we don't get info within the time frame, break the loop
	}
	return 1;
}
void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients) {
	Mat frame, drawToFrame; 
	vector<Mat> savedImages;//vector of Mat objects; save good calibrations
	vector<vector<Point2f>>markerCorners, rejectedCandidates; //vector of vectors of point2f objects
	VideoCapture vid(0);
	if (!vid.isOpened()) { return ; }
	int framesPerSecond = 20;
	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
	while (true) {
		if (!vid.read(frame)) //if you can't read a frame, break
			break;
		vector < Vec2f > foundPoints;  //vector to store all found pts
		bool found = false;
		found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame); //copy the frame to another Mat, over which we will overlay
		drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);//draw chessboard corners if we find them
		if (found)
			imshow("Webcam", drawToFrame);
		else
			imshow("Webcam", frame);
		char character = waitKey(1000 / framesPerSecond);
		switch (character) {
		case ' ': //space key
			if (found) { //saving img
				Mat temp; frame.copyTo(temp); savedImages.push_back(temp);
			}
			break;
		case 13: //enter key
			if (savedImages.size() > 15) {//check if we have enough valid image before calibration
				cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);//start calibration
				saveCameraCalibration("Camera calibration", cameraMatrix, distanceCoefficients);
			}
			break;
		case 27:
			//exit
			return ; break;
		}
	}
}
int main(int argv, char** argc) {
	//createArucoMarker(); //running this function by itself the first time filled the folder with 50 aruco markers
	Mat distanceCoefficients, cameraMatrix = Mat::eye(3, 3, CV_64F); //curent video frame; camera matrix starting as an identity matrix
	//cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
	loadCameraCalibration("Camera calibration",cameraMatrix,distanceCoefficients);
	startWebcamMonitoring(cameraMatrix,distanceCoefficients,0.088f);
	return 0;
}
//Camera calibration involves going from real world units (m) to camera units (px); Fix radial and tangential distortions
//Tangential distortion occurs due to the image-taking lenses not being perfectly parallel to the imaging plane; Both give 5 distortion params
//THey are the camera focal lengths fx and fy, optical centers (in px) cx and cy and focal length f.
//Determining the distortion and camera matrices is known as calibration