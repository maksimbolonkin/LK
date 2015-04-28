
#include "utils.hxx"



int main(int argc, char *argv[])
{

	if(argc<3)
	{
		cout<<"Enter two filenames: fixed image and moving image."<<endl;
		exit(-1);
	}

	//Mat fixed = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	//fixed.convertTo(fixed, CV_64FC1);
	Mat moving = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	moving.convertTo(moving, CV_64FC1);
	// read the images I J

	//Point2f src[3]; src[0] = Point2f(50, 50); src[1] = Point2f(200, 50); src[2] = Point2f(50, 200); 
	//Point2f dst[3]; dst[0] = Point2f(40, 40); dst[1] = Point2f(200, 40); dst[2] = Point2f(180, 60); 
	//Mat X = getAffineTransform(src, dst);
	Mat X = getRotationMatrix2D(Point2f(190, 100), -35, 0.9);
	cout<<X<<endl;
	Mat fixed;
	warpAffine(moving, fixed, X, moving.size());
	imwrite("genfixed.jpeg", fixed);

	cout<<"Images have been read..."<<endl;

	Mat aff = regImage(fixed, moving);

	cout<<aff<<endl;
	Mat newimg;
	warpAffine(fixed, newimg, aff, moving.size());

	imwrite("new.jpeg", newimg);

}
