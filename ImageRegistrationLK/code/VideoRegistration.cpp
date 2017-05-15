#include "ImageRegistrationLK.h"
#include "utils.hxx"
#include <fstream>
#include <stdio.h>

int main(int argc, char *argv[])
{
	ofstream fout("result.txt");

	char path[100];

	double error = 0;
	
	int wx = 312, wy = 154;
	Point2d offset(539, 138);

	for(int i=1; i<25; i++)
	{
		cout<<i<<" -> "<<i+1<<endl;

		sprintf(path, "my_video2/%d.jpg", i);
		Mat fixed = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		fixed.convertTo(fixed, CV_64FC1);

		sprintf(path, "my_video2/%d.jpg", i+1);
		Mat moving = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		moving.convertTo(moving, CV_64FC1);

		ImageRegistrationLK reg;
		reg.setFixedImage(fixed);
		reg.setMovingImage(moving);
		if(MIN(wx,wy)<100)
			reg.setNumberOfLevels(2);
		else if(MIN(wx,wy)<40)
			reg.setNumberOfLevels(1);
		else
			reg.setNumberOfLevels(3);
		//reg.setMaxIterations(100);
		reg.setWindowSize( Size(wx,wy));
		reg.setWindowOffset(offset);
		reg.setRegistrationMode(REG_UNIFORM_SCALE|REG_SHIFT);
		//reg.setMask(mask);
		reg.runRegistration();

		Mat aff = reg.getTransform();
		fout<<i<<" -> "<<i+1<<endl;
		//fout<<aff<<endl<<endl;

		char out[100];

		{
			sprintf(out, "masks/mask-%d-%d.jpeg", i, i+1);
			Mat mask = reg.getMask();
			imwrite(out, mask);
		}
		
		{
			sprintf(out, "difference/plate-diff-%d-%d.jpeg", i, i+1);
			Mat newimg;
			Mat aff1; 
			aff.copyTo(aff1);
			aff1.at<double>(0,2) += offset.x - aff1.at<double>(0,0)*offset.x - aff1.at<double>(0,1)*offset.y;
			aff1.at<double>(1,2) += offset.y - aff1.at<double>(1,0)*offset.x - aff1.at<double>(1,1)*offset.y;
			warpAffine(moving, newimg, aff1, moving.size(), WARP_INVERSE_MAP);
			Mat diff = abs(fixed-newimg);
			int row1 = int(offset.y), row2 = int(offset.y+wy)<moving.rows?int(offset.y+wy):moving.rows-1;
			int col1 = int(offset.x), col2 = int(offset.x+wx)<moving.cols?int(offset.x+wx):moving.cols-1;
			imwrite(out, diff(Range(row1, row2), Range(col1, col2)));
			fout<<mean(diff);
			error += mean(diff)[0];
			//imwrite(out, reg.getDifference());
		}

		{
			// transform offset for next im
			double alpha = aff.at<double>(0,0), beta = aff.at<double>(1,1);
			wx *= alpha; wy *= beta;
			offset.x += aff.at<double>(0,2); offset.y += aff.at<double>(1,2);	

			sprintf(out, "tracking/%d.jpeg", i);
			int row1 = int(offset.y), row2 = int(offset.y+wy)<moving.rows?int(offset.y+wy):moving.rows-1;
			int col1 = int(offset.x), col2 = int(offset.x+wx)<moving.cols?int(offset.x+wx):moving.cols-1;
			imwrite(out, moving(Range(row1, row2), Range(col1, col2)));

			sprintf(out, "tracking/rect%d.jpeg", i);
			rectangle(moving, Point(col1, row1), Point(col2, row2), 255.0, 3);
			imwrite(out, moving);

		}
	}
	fout<<error;
	fout.close();
}
