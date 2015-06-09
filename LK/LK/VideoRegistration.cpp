#include "ImageRegistrationLK.h"
#include "utils.hxx"
#include <fstream>
#include <stdio.h>


int main(int argc, char *argv[])
{
	ofstream fout("result.txt");

	char path[100];
	
	int wx = 290, wy = 98;
	Point2d offset(1601, 424);

	for(int i=7; i<15; i++)
	{
		cout<<i<<" -> "<<i+1<<endl;

		sprintf(path, "accident_video/%d.bmp", i);
		Mat fixed = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		fixed.convertTo(fixed, CV_64FC1);

		sprintf(path, "accident_video/%d.bmp", i+1);
		Mat moving = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		moving.convertTo(moving, CV_64FC1);

		ImageRegistrationLK reg;
		reg.setFixedImage(fixed);
		reg.setMovingImage(moving);
		reg.setNumberOfLevels(2);
		//reg.setMaxIterations(100);
		reg.setWindowSize( Size(wx,wy));
		reg.setWindowOffset(offset);
		reg.setRegistrationMode(REG_UNIFORM_SCALE|REG_SHIFT);
		//reg.setMask(mask);
		reg.runRegistration();

		Mat aff = reg.getTransform();
		fout<<i<<" -> "<<i+1<<endl;
		fout<<aff<<endl<<endl;

		char out[100];
		sprintf(out, "plate-diff-%d-%d.jpeg", i, i+1);
		Mat newimg;
		aff.at<double>(0,2) += offset.x - aff.at<double>(0,0)*offset.x - aff.at<double>(0,1)*offset.y;
		aff.at<double>(1,2) += offset.y - aff.at<double>(1,0)*offset.x - aff.at<double>(1,1)*offset.y;
		warpAffine(moving, newimg, aff, moving.size(), WARP_INVERSE_MAP);
		//imwrite(out, abs(fixed-newimg));
		imwrite(out, reg.getDifference());

		double alpha = aff.at<double>(0,0);
		wx *= alpha; wy *=alpha;
		offset.x += aff.at<double>(0,2); offset.y += aff.at<double>(1,2);	//???????
	}

	fout.close();
}
