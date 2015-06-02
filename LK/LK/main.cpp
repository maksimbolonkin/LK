#include "ImageRegistrationLK.h"
#include "utils.hxx"
#include <fstream>


int main(int argc, char *argv[])
{
	ofstream fout("result.txt");


	Mat bg = imread("bg6.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	bg.convertTo(bg, CV_64FC1);
	Mat pattern = imread("pattern3.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	pattern.convertTo(pattern, CV_64FC1);
	Mat mask = formPatternMask(pattern);
	//imwrite("mask.jpeg", mask);

	// resize pattern
	Mat newpat;
	pattern.
	resize(pattern, newpat, Size(0,0), 1.0, 1.0, INTER_LINEAR);
	pattern.copyTo(newpat);

	Mat fixed = formImage(bg, pattern, mask, Point2d(100, 100));
	Mat moving = formImage(bg, newpat, mask, Point2d(125, 150));
	//Mat fixed = imread("real1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//fixed.convertTo(fixed, CV_64FC1);
	//Mat moving = imread("real2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//moving.convertTo(moving, CV_64FC1);

	//int wx = MAX(pattern.size().width, newpat.size().width), wy = MAX(pattern.size().height, newpat.size().height);
	int wx = 215, wy = 249;

	imwrite("1.jpeg", fixed);
	imwrite("2.jpeg", moving);


	//cout<<"Images have been read..."<<endl;

	ImageRegistrationLK reg;
	reg.setFixedImage(fixed);
	reg.setMovingImage(moving);
	reg.setNumberOfLevels(4);
	reg.setMaxIterations(100);
	reg.setWindowSize( Size(wx,wy));
	reg.setWindowOffset(Point2d(397,174));
	//reg.setMask(mask);
	//reg.runRegistration();

	Mat aff = reg.getTransform();

	fout<<aff<<endl;
	Mat newimg;
	warpAffine(fixed, newimg, aff, moving.size());

	//Mat diffIm = reg.markPatternAndBackground(fixed, moving, aff, Point2d(100,100),  Size(wx,wy));

	//reg.setMask(diffIm);
	//reg.runRegistration();
	
	//imwrite("after-3x3-calculating.jpeg", diffIm);
	//threshold(diffIm, diffIm, 1.0, 255.0, THRESH_BINARY);
	//imwrite("after-thresholding.jpeg", diffIm);
	//diffIm.convertTo(diffIm, CV_8UC1);
	//equalizeHist(diffIm, diffIm);
	//imwrite("diff.jpeg", diffIm);

	//patternSegmentation(diffIm, 2);

	Mat diff=reg.getDifference();
	imwrite("diff.jpeg", diff);

	imwrite(argv[3], newimg);

	fout.close();
}
