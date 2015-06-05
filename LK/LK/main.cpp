#include "ImageRegistrationLK.h"
#include "utils.hxx"
#include <fstream>


int main(int argc, char *argv[])
{
	ofstream fout("result.txt");

	//----- read bg and pattern, form image

	Mat bg = imread("bg5.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	bg.convertTo(bg, CV_64FC1);
	Mat pattern = imread("pattern3.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	pattern.convertTo(pattern, CV_64FC1);
	Mat mask = formPatternMask(pattern);
	imwrite("mask.jpeg", mask);
	// resize pattern
	Mat newpat;
	////Mat X = getRotationMatrix2D(Point2f(pattern.size().width/2, pattern.size().height/2), 17, 0.85);
	////warpAffine(pattern, newpat, X,Size(0,0));
	resize(pattern, newpat, Size(0,0), 1.15, 1.15, INTER_LINEAR);
	////pattern.copyTo(newpat);
	Mat fixed = formImage(bg, pattern, mask, Point2d(100, 100));
	Mat moving = formImage(bg, newpat, formPatternMask(newpat), Point2d(97, 104));
	imwrite("1.jpeg", fixed);
	imwrite("2.jpeg", moving);
	int wx = pattern.size().width, wy = pattern.size().height;
	Point2d offset(100, 100);

	//----- read from file

	//Mat fixed = imread("fixed.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//fixed.convertTo(fixed, CV_64FC1);
	//Mat moving = imread("moving.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//moving.convertTo(moving, CV_64FC1);
	//int wx = 284, wy = 76;
	//Point2d offset(1438, 421);

	


	//cout<<"Images have been read..."<<endl;

	ImageRegistrationLK reg;
	reg.setFixedImage(fixed);
	reg.setMovingImage(moving);
	reg.setNumberOfLevels(4);
	//reg.setMaxIterations(100);
	reg.setWindowSize( Size(wx,wy));
	reg.setWindowOffset(offset);
	reg.setRegistrationMode(REG_UNIFORM_SCALE|REG_SHIFT);
	//reg.setMask(mask);
	reg.runRegistration();

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
