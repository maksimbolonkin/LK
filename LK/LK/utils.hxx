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
	for(int x = 0; x < wx; x++)
	{
		for(int y = 0; y< wy; y++)
		{
			Vec2d gI = I.grad(x,y);
			double Ix = gI[0]*sin(CV_PI*x/(wx-1)), Iy = gI[1]*sin(CV_PI*y/(wy-1));
			Mat D2 = (Mat_<double>(6,1) << Ix, Iy, x*Ix, y*Ix, x*Iy, y*Iy);
			G += D2*D2.t();			
		}
	}
	return G;
}

Mat calcDiffVector(Image I, Image J, int wx, int wy)
{
	Mat b = Mat::zeros(6,1,CV_64FC1);
	//Mat Gx = getGaussianKernel(wx, -1), Gy = getGaussianKernel(wy,-1);
	//Mat G = Gy*Gx.t();
	for(int x = 0; x < wx; x++)
	{
		for(int y = 0; y < wy; y++)
		{
			double delta = (I.at(x,y) - J.at(x,y))*sin(CV_PI*x/(wx-1))*sin(CV_PI*y/(wy-1));

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

Mat regImage(Mat fixed, Mat moving)
{
	int MaxIter = 25;	// could be set by user
	double eps = 0.00001;	// could be set by user

	//creating pyramids
	Mat PyramidI[5], PyramidJ[5];
	PyramidI[0] = fixed; PyramidJ[0] = moving;
	for(int i=1; i<5; i++)
	{
		PyramidI[i] = GetPyramidNextLevel(PyramidI[i-1]);
		PyramidJ[i] = GetPyramidNextLevel(PyramidJ[i-1]);
		cout<<"Pyramid level "<<i<<" size: "<<PyramidI[i].size().width<<"x"<<PyramidI[i].size().height;
		cout<<" and "<< PyramidJ[i].size().width<<"x"<<PyramidJ[i].size().width<<endl;
	}

	cout<<"Pyramids created...";
	// init global guesses vg Ag
	Mat vg = Mat::zeros(2,1, CV_64FC1);
	Mat Ag = Mat::eye(2,2,CV_64FC1);

	
	Mat A(Ag); Mat v(vg);
	// init local guesses v A (for the least precise pyramid level)
	cout<<"Init v = "<< v<<endl<<"A = "<< A <<endl;

	for(int L = 4; L>=0; L--)
	{
		cout<<"----- Level "<<L<<" -----"<<endl;
		//cout<<PyramidI[L]<<endl<<PyramidJ[L]<<endl;
		Image I(PyramidI[L]);
		Image J(PyramidJ[L]);

		// window size can be adjusted if necessary, whole image by default (-1 for purpose of gradient calculation)
		int wx = PyramidI[L].size().width, wy = PyramidI[L].size().height;

		//translate image I (to do it centralized at the point of interest -- may be as in ITK centralize to mass center)
		//Moments mI = moments(PyramidI[L]);
		//Vec2d uI( int(mI.m10/mI.m00), int(mI.m01/mI.m00));		// -- mass center to centralize fixed image, also a point of interest, since there's no
												// feature point for entire picture or some region
		//Vec2d uI(PyramidI[L].size().width/2-1, PyramidI[L].size().height/2-1);
		//I.Centralise(uI);
		//cout<<"Center for I = "<< uI<<endl;

		//Moments mJ = moments(PyramidJ[L]);
		//Vec2d uJ( int(mJ.m10/mJ.m00), int(mJ.m01/mJ.m00));
		//Vec2d uJ(PyramidJ[L].size().width/2-1, PyramidJ[L].size().height/2-1);
		//J.Centralise(uJ);
		//cout<<"Center for J = "<< uJ<<endl;

		cout<<"Calculating G (grad matrix)..."<<endl;
		Mat G = calcGradMatrix(I, wx, wy);
		// compute G - gradient matrix (?? matrix operations maybe)
		cout<<"G = "<<endl<<G<<endl;
		
		//guess from previous level of pyramid
		v = 2*v;
		cout<<"Guess for v = "<<v<<endl;

		int iter = 0;
		while(true)
		{

			J.setTransform(A, v);
			//warp second image

			Mat b = calcDiffVector(I, J, wx, wy);
			cout<<"Diff vector = "<<b<<endl;

			Mat v_opt = G.inv(DECOMP_SVD)*b;
			cout<<"SVD solution = "<<v_opt<<endl;

			Mat tA(2,2,CV_64FC1);
			Mat tv(2,1,CV_64FC1);
			
			tv.at<double>(0,0) = v_opt.at<double>(0,0);
			tv.at<double>(1,0) = v_opt.at<double>(1,0);
			tA.at<double>(0,0) = 1+v_opt.at<double>(2,0); tA.at<double>(0,1) = v_opt.at<double>(3,0);
			tA.at<double>(1,0) = v_opt.at<double>(4,0); tA.at<double>(1,1) = 1+v_opt.at<double>(5,0);

			v = v + A*tv;
			A = A*tA;
			
			cout<<"v = "<<v<<endl;
			cout<<"A = "<<endl<<A<<endl;

			iter++;
			if(iter>= MaxIter)
				break;
			double shift = sqrt(tv.dot(tv));
			cout<<"Shift = "<<shift<<endl;
			if(shift < eps)
				break;
			
		}

	}	// end of pyramid loop

	Mat aff(2,3,CV_64FC1);
	aff.at<double>(0,0) = A.at<double>(0,0); aff.at<double>(0,1) = A.at<double>(0,1);
	aff.at<double>(1,0) = A.at<double>(1,0); aff.at<double>(1,1) = A.at<double>(1,1);
	aff.at<double>(0,2) = v.at<double>(0,0); aff.at<double>(1,2) = v.at<double>(1,0);

	return aff;
}