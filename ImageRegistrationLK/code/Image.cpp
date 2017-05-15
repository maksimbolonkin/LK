#include "Image.h"

Point2d Image::calcPoint(double x, double y)
{
	Mat pt0 = (Mat_<double>(2,1) << x, y);
	Mat pt1 = _A*pt0 + _d + _c;
	return Point2d(pt1.at<double>(0,0), pt1.at<double>(1,0));
}

double Image::getImgPixel(int x, int y)
{
	if((0<= x) && (x <im.cols) && (0<= y) && (y <im.rows))
		return im.at<double>(y,x);
	else
		return 0;
}

double Image::interpolate(Point2d p)
{
	// LeftBottom = x1y1, LU = x1y2, RB = x2y1, RU = x2y2
	double x1 = floor(p.x); double x2 = ceil(p.x);
	double y1 = floor(p.y); double y2 = ceil(p.y);
	// bilinear interpolation
	double r1 = abs(p.x - x1)<0.00001? getImgPixel(x1,y1) : (x2-p.x)*getImgPixel(x1,y1) + (p.x-x1)*getImgPixel(x2,y1);
	double r2 = abs(p.x - x1)<0.00001? getImgPixel(x1,y2) : (x2-p.x)*getImgPixel(x1,y2) + (p.x-x1)*getImgPixel(x2,y2);
	double r = abs(p.y - y1)<0.00001? r1 : (y2-p.y)*r1 + (p.y-y1)*r2;
	return r;
}

Image::Image(Mat img)
{
	im = Mat(img);
	_A = Mat::eye(2,2,CV_64FC1);
	_d = Mat::zeros(2,1,CV_64FC1);
	_c = Mat::zeros(2,1,CV_64FC1);
}

void Image::setImage(Mat img)
{
	im = img;
}

void Image::Centralise(Vec2d center)
{
	_c = Mat(center);
}

void Image::setTransform(Mat A, Mat d)
{
	_A = A;
	_d = d;
}

double Image::at(int x, int y)
{
	return interpolate(calcPoint(x,y));
}

Vec2d Image::grad(int x, int y)
{
	double x1 = at(x+1, y), x2 = at(x-1, y);
	double dx = (x1 - x2)/2;
	double y1 = at(x, y+1), y2 = at(x, y-1);
	double dy = (y1 - y2)/2;
	return Vec2d(dx, dy);
}

Mat Image::getImage()
{return im;}