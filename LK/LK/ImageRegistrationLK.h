#pragma once

#include "Image.h"
#include <vector>

using namespace std;

class ImageRegistrationLK
{
private:
	int MaxIter;	// could be set by user
	double eps;	// could be set by user
	int NumOfLevels;
	Mat fixed, moving, mask;
	Mat affTransform;
	Size window;
	Point2d offset;
	bool _isMaskSet;

	class WindowFunction
	{
	private:
		Mat windowMask;
		double Rmax;

	public:
		WindowFunction(){};
		void setMask(Mat m, int wx, int wy);
		void setDefaultMask(Size sz);
		double getValue(int x, int y);
	};
	WindowFunction w;

	// auxillary funcs
	Mat calcGradMatrix(Image I, int wx, int wy);
	Mat calcDiffVector(Image I, Image J, int wx, int wy);
	double calcSquareDifference(Image I, Image J, int wx, int wy);
	
	double getPixel(Mat I, int i, int j);
	Mat GetPyramidNextLevel(Mat I);

public:
	ImageRegistrationLK();

	void setMaxIterations(int it);
	void setPrecision(double it);
	void setNumberOfLevels(int it);
	void setFixedImage(Mat f);
	void setMovingImage(Mat f);
	void setWindowSize(Size w);
	void setWindowOffset(Point2d pos);
	void setMask(Mat m);
	Mat getTransform();

	void runRegistration();

	Mat markPatternAndBackground(Mat fixed, Mat moving, Mat aff, Point2d pos, Size window);
};