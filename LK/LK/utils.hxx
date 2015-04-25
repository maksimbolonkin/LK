#include <cv.h>
#include <iostream>
#include <highgui.h>
#include <math.h>
#include <calib3d\calib3d.hpp>


using namespace cv;
using namespace std;

class Image
{
private:
	Mat im;
	Mat _c;	// center to centralize
	Mat _A;		// affine transform
	Mat _d;	// translation vector

	Point2d calcPoint(double x, double y)
	{
		Mat pt0 = (Mat_<double>(2,1) << x, y);
		Mat pt1 = _A*pt0 + _d + _c;
		return Point2d(pt1.at<double>(0,0), pt1.at<double>(1,0));
	}

	double getImgPixel(int x, int y)
	{
		if((0<= x) && (x <im.cols) && (0<= y) && (y <im.rows))
			return im.at<double>(y,x);
		else
			return 0;
	}

	double interpolate(Point2d p)
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

public:
	Image(Mat img)
	{
		im = Mat(img);
		_A = Mat::eye(2,2,CV_64FC1);
		_d = Mat::zeros(2,1,CV_64FC1);
		_c = Mat::zeros(2,1,CV_64FC1);
	}

	void setImage(Mat img)
	{
		im = img;
	}

	void Centralise(Vec2d center)
	{
		_c = Mat(center);
	}

	void setTransform(Mat A, Mat d)
	{
		_A = A;
		_d = d;
	}
	
	double at(int x, int y)
	{
		return interpolate(calcPoint(x,y));
	}

	Vec2d grad(int x, int y)
	{
		double x1 = at(x+1, y), x2 = at(x-1, y);
		double dx = (x1 - x2)/2;
		double y1 = at(x, y+1), y2 = at(x, y-1);
		double dy = (y1 - y2)/2;
		return Vec2d(dx, dy);
	}

	Mat getImage()
	{return im;}
};

Mat calcGradMatrix(Image I, int wx, int wy)
{
	Mat G = Mat::zeros(6,6,CV_64FC1);
	Mat temp = I.getImage();
	for(int x = -wx; x<= wx; x++)
	{
		for(int y = -wy; y<=wy; y++)
		{
			Vec2d gI = I.grad(x,y);
			double Ix = gI[0], Iy = gI[1];
			Mat D2 = (Mat_<double>(6,1) << Ix, Iy, x*Ix, y*Ix, x*Iy, y*Iy);
			G += D2*D2.t();			
		}
	}
	return G;
}

Mat calcDiffVector(Image I, Image J, int wx, int wy)
{
	Mat b = Mat::zeros(6,1,CV_64FC1);
	for(int x = -wx; x<= wx; x++)
	{
		for(int y = -wy; y<=wy; y++)
		{
			double delta = I.at(x,y) - J.at(x,y);

			Vec2d gI = I.grad(x,y);
			double Ix = gI[0], Iy = gI[1];
			Mat D2 = (Mat_<double>(6,1) << Ix, Iy, x*Ix, y*Ix, x*Iy, y*Iy)*delta;
			
			b += D2;
		}
	}
	return b;
}

double getPixel(Mat I, int i, int j)
{
	if(i==-1)
		i = 0;
	if(j==-1)
		j=0;
	if(i>=I.rows)
		i = I.rows-1;
	if(j>=I.cols)
		j = I.cols-1;
	double res = I.at<double>(i,j);
	return res;
}

Mat GetPyramidNextLevel(Mat I)
{
	int cols = (I.cols+1)/2, rows = (I.rows+1)/2;
	Mat res(rows, cols, CV_64FC1);
	for(int i=0; i<rows; i++)
		for(int j=0; j<cols; j++)
		{
			res.at<double>(i,j) = (int)(getPixel(I, 2*i, 2*j)/4.0 +
				(getPixel(I, 2*i-1, 2*j)+getPixel(I, 2*i+1, 2*j)+getPixel(I, 2*i, 2*j-1)+getPixel(I, 2*i, 2*j+1))/8.0+
				(getPixel(I, 2*i-1, 2*j-1)+getPixel(I, 2*i-1, 2*j+1)+getPixel(I, 2*i+1, 2*j-1)+getPixel(I, 2*i+1, 2*j+1))/16);
		}
	return res;
}

