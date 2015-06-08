#include "ImageRegistrationLK.h"
#include "utils.hxx"
#include <fstream>
#include <stdio.h>


int main(int argc, char *argv[])
{
	ofstream fout("result.txt");

	char path[100];
	
	int wx = 181, wy = 33;
	Point2d offset(903, 426);

	ImageRegistrationLK reg;
	
	for(int i=1; i<10; i++)
	{
		sprintf(path, "my_video/%d.jpg", i);
		Mat fixed = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		fixed.convertTo(fixed, CV_64FC1);

		sprintf(path, "my_video/%d.jpg", i+1);
		Mat moving = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		moving.convertTo(moving, CV_64FC1);

		
		reg.setFixedImage(fixed);
		reg.setMovingImage(moving);
		reg.setNumberOfLevels(3);
		//reg.setMaxIterations(100);
		reg.setWindowSize( Size(wx,wy));
		reg.setWindowOffset(offset);
		reg.setRegistrationMode(REG_UNIFORM_SCALE|REG_SHIFT);
		//reg.setMask(mask);
		reg.runRegistration();

		Mat aff = reg.getTransform();
		fout<<i<<" -> "<<i+1<<endl;
		fout<<aff<<endl<<endl;

		double alpha = aff.at<double>(0,0);
		wx *= alpha; wy *=alpha;
		offset.x += aff.at<double>(0,2); offset.y += aff.at<double>(1,2);	//???????

		char out[100];
		sprintf(out, "diff-%d-%d.jpeg", i, i+1);
		imwrite(out, reg.getDifference());
	}

	fout.close();
}
