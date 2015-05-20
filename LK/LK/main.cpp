
#include "utils.hxx"
#include <fstream>


int main(int argc, char *argv[])
{
	ofstream fout("result.txt");


	Mat bg = imread("background.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	bg.convertTo(bg, CV_64FC1);
	Mat pattern = imread("pattern.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	pattern.convertTo(pattern, CV_64FC1);

	// resize pattern
	Mat newpat;
	resize(pattern, newpat, Size(0,0), 1.0, 1.0, INTER_LINEAR);

	Mat fixed = formImage(bg, pattern, Point2d(100, 100));
	Mat moving = formImage(bg, newpat, Point2d(105, 100));

	int wx = MAX(pattern.size().width, newpat.size().width), wy = MAX(pattern.size().height, newpat.size().height);

	imwrite("1.jpeg", fixed);
	imwrite("2.jpeg", moving);


	//cout<<"Images have been read..."<<endl;

	ImageRegistrationLK reg;
	reg.setFixedImage(fixed);
	reg.setMovingImage(moving);
	reg.setNumberOfLevels(3);
	reg.setWindowSize( Size(wx,wy));
	reg.setWindowOffset(Point2d(100,100));
	reg.runRegistration();

	Mat aff = reg.getTransform();

	fout<<aff<<endl;
	Mat newimg;
	warpAffine(fixed, newimg, aff, moving.size());

	imwrite(argv[3], newimg);

	fout.close();
}
