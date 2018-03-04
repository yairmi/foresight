// foresight.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>
#include "disparityMap.h"
using namespace std::chrono;
using namespace cv;
const char *windowDisparityTitle = "Disparity Image";

int main()
{
	auto imageLeft = imread("left_rect.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	auto imageRight = imread("right_rect.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	if (!imageLeft.data)                              // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	if (!imageRight.data)                              // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat resultDisparityMap(imageLeft.rows, imageLeft.cols, CV_8UC1);
	disparityMap dm(130, 5);
	//Get current time in milliseconds
	auto ms = duration_cast<milliseconds>(
		system_clock::now().time_since_epoch());
	std::cout << "start generate Disparity image" << std::endl;
	dm.generateDisparityMapParallel(imageLeft, imageRight, resultDisparityMap);
	//Note !!! The following comment is the sequesntial implementation for calculatin disparity image
	//dm.generateDisparityMapSeq(imageLeft, imageRight, resultDisparityMap);
	std::cout << (duration_cast<milliseconds>(
		system_clock::now().time_since_epoch()) - ms).count();
	namedWindow(windowDisparityTitle, WINDOW_NORMAL);
	imshow(windowDisparityTitle, resultDisparityMap);

	waitKey(0);

	imwrite("disparityResult.png", resultDisparityMap);
	return 0;
}