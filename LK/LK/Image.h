#include <opencv\cv.h>
#include <iostream>
#include <opencv\highgui.h>
#include <math.h>
#include <calib3d\calib3d.hpp>

using namespace cv;

class Image
{
private:
	Mat im;
	Mat _c;	// upper-left corner of a window of interest
	Mat _A;		// affine transform
	Mat _d;	// translation vector

	Point2d calcPoint(double x, double y);
	double getImgPixel(int x, int y);
	double interpolate(Point2d p);

public:
	Image(Mat img);
	void setImage(Mat img);
	void Centralise(Vec2d center);
	void setTransform(Mat A, Mat d);
	double at(int x, int y);
	Vec2d grad(int x, int y);
	Mat getImage();
};