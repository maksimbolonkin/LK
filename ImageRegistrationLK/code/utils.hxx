#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <opencv2/calib3d.hpp>

using namespace cv;

//--------------------
// funcs for generating fixed and moved images from background and pattern

Mat normalizeBackground(Mat bg, double low, double high)
{
	Mat res;
	normalize(bg, res, low, high, NORM_MINMAX);
	return res;
}

Mat formPatternMask(Mat p)
{
	Mat res;
	p.convertTo(p, CV_32FC1);
	threshold(p, res, 200,255,THRESH_BINARY);
	res.convertTo(res, CV_64FC1);
	return res;
}

Mat formImage(Mat bg, Mat pattern, Mat mask, Point2d pos)
{
	Mat res = normalizeBackground(bg, 128.0, 255.0);

	for(int i = 0; i<pattern.size().width; i++)
		for(int j=0; j<pattern.size().height; j++)
			if(mask.at<double>(j,i) < 50)
				res.at<double>(pos.y+j, pos.x+i) = pattern.at<double>(j,i);
	return res;
}


