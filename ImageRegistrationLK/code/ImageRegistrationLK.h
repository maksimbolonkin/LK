#pragma once

#include "Image.h"
#include <vector>

using namespace std;

// Registration modes
enum
{
	REG_SHIFT = 1,
	REG_SCALE = 2,
	REG_ROTATION = 4,
	REG_SCEW = 8,
	REG_UNIFORM_SCALE = 16,
	REG_AFFINE = 32
};


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
	int _isMaskSet;		// 0 - no mask, 1 - default (cos) mask, 2 - custom mask
	int regMode;

	class WindowFunction
	{
	private:
		Mat windowMask;
		double Rmax;

	public:
		bool nomask;
		WindowFunction(){ nomask = false; };
		void setMask(Mat m, int wx, int wy);
		void setDefaultMask(Size sz);
		double getValue(int x, int y);
		Mat getMask();
	};
	WindowFunction w;

	// auxillary funcs
	Mat calcGradMatrix(Image I, int wx, int wy);
	Mat calcDiffVector(Image I, Image J, int wx, int wy);
	double calcSquareDifference(Image I, Image J, int wx, int wy);
	
	double getPixel(Mat I, int i, int j);
	Mat GetPyramidNextLevel(Mat I);

	Mat getParametrisationMatrix();
	Mat interpretResult_shift(Mat v_opt);
	Mat interpretResult_affine(Mat v_opt);

public:
	ImageRegistrationLK();

	void setMaxIterations(int it);
	void setPrecision(double it);
	void setNumberOfLevels(int it);
	void setFixedImage(Mat f);
	void setMovingImage(Mat f);
	void setWindowSize(Size w);
	void setWindowOffset(Point2d pos);
	void setRegistrationMode(int _regMode);
	void setMask(Mat m);
	void setMask(bool mask);
	Mat getMask();
	Mat getTransform();

	void runRegistration();

	Mat markPatternAndBackground(Mat fixed, Mat moving, Mat aff, Point2d pos, Size window);
	Mat getDifference();
};
