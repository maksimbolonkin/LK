#include "ImageRegistrationLK.h"

void ImageRegistrationLK::WindowFunction::setMask(Mat m, int wx , int wy) 
{ 
	Mat temp;
	// resize to fit appropriate pyramid level
	resize(m, m, Size(0,0), (double)wx/m.size().width, (double)wy/m.size().height, CV_INTER_LINEAR);
	m.convertTo(m, CV_32FC1);
	threshold(m, temp, 0.5, 255, THRESH_BINARY_INV);
	temp.convertTo(temp, CV_8UC1);
	distanceTransform(temp, windowMask, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	windowMask.convertTo(windowMask, CV_64FC1);

	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc( windowMask, &minVal, &maxVal, &minLoc, &maxLoc );
	Rmax = maxVal;

	//Mat nim;
	//normalize(windowMask, nim,0.0, 255.0, NORM_MINMAX);
	//imwrite("m1.jpeg", nim);
}

void ImageRegistrationLK::WindowFunction::setDefaultMask(Size sz)
{
	windowMask = Mat(sz, CV_64FC1);
	for(int i=0; i<sz.width;i++)
		for(int j=0; j<sz.height; j++)
			windowMask.at<double>(j,i) = sqrt((i-sz.width/2)*(i-sz.width/2) + (j-sz.height/2)*(j-sz.height/2));

	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc( windowMask, &minVal, &maxVal, &minLoc, &maxLoc );
	Rmax = maxVal;

	windowMask = Mat::ones(sz, CV_64FC1)*Rmax - windowMask;

	//Mat nim;
	//normalize(windowMask, nim,0.0, 255.0, NORM_MINMAX);
	//imwrite("m1.jpeg", nim);
}

double ImageRegistrationLK::WindowFunction::getValue(int x, int y) 
{ 
	double R = windowMask.at<double>(y,x);
	return (1-cos(CV_PI*R/Rmax));
}

Mat ImageRegistrationLK::calcGradMatrix(Image I, int wx, int wy)
{
	Mat G = Mat::zeros(6,6,CV_64FC1);
	Mat temp = I.getImage();
	for(int x = 0; x < wx; x++)
	{
		for(int y = 0; y< wy; y++)
		{
			Vec2d gI = I.grad(x,y);
			double Ix = gI[0]*sqrt(w.getValue(x,y)), Iy = gI[1]*sqrt(w.getValue(x,y));
			Mat D2 = (Mat_<double>(6,1) << Ix, Iy, x*Ix, y*Ix, x*Iy, y*Iy);
			G += D2*D2.t();			
		}
	}
	return G;
}
Mat ImageRegistrationLK::calcDiffVector(Image I, Image J, int wx, int wy)
{
	Mat b = Mat::zeros(6,1,CV_64FC1);
	//Mat Gx = getGaussianKernel(wx, -1), Gy = getGaussianKernel(wy,-1);
	//Mat G = Gy*Gx.t();
	for(int x = 0; x < wx; x++)
	{
		for(int y = 0; y < wy; y++)
		{
			double delta = (I.at(x,y) - J.at(x,y))*w.getValue(x,y);

			Vec2d gI = I.grad(x,y);
			double Ix = gI[0], Iy = gI[1];
			Mat D2 = (Mat_<double>(6,1) << Ix, Iy, x*Ix, y*Ix, x*Iy, y*Iy)*delta;

			b += D2;
		}
	}
	return b;
}

double ImageRegistrationLK::getPixel(Mat I, int i, int j)
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
Mat ImageRegistrationLK::GetPyramidNextLevel(Mat I)
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

ImageRegistrationLK::ImageRegistrationLK()
{
	MaxIter = 25;
	eps = 0.00001;	
	NumOfLevels = 2;
	affTransform = Mat(2,3,CV_64FC1);
}

void ImageRegistrationLK::setMaxIterations(int it) { MaxIter = it; }
void ImageRegistrationLK::setPrecision(double it) { eps = it; }
void ImageRegistrationLK::setNumberOfLevels(int it) { NumOfLevels = it; }
void ImageRegistrationLK::setFixedImage(Mat f) { f.copyTo(fixed); }
void ImageRegistrationLK::setMovingImage(Mat f) { f.copyTo(moving); }
void ImageRegistrationLK::setWindowSize(Size w) { window = w; }
void ImageRegistrationLK::setWindowOffset(Point2d pos) { offset = pos; }
Mat ImageRegistrationLK::getTransform() { return affTransform; }

void ImageRegistrationLK::runRegistration()
{
	//creating pyramids
	vector<Mat> PyramidI, PyramidJ;
	PyramidI.push_back(fixed); PyramidJ.push_back(moving);
	//Mat PyramidI[NumOfLevels], PyramidJ[NumOfLevels];
	//PyramidI[0] = fixed; PyramidJ[0] = moving;
	for(int i=1; i<NumOfLevels; i++)
	{
		/*PyramidI[i] = GetPyramidNextLevel(PyramidI[i-1]);
		PyramidJ[i] = GetPyramidNextLevel(PyramidJ[i-1]);*/
		PyramidI.push_back(GetPyramidNextLevel(PyramidI[i-1]));
		PyramidJ.push_back(GetPyramidNextLevel(PyramidJ[i-1]));
		cout<<"Pyramid level "<<i<<" size: "<<PyramidI[i].size().width<<"x"<<PyramidI[i].size().height;
		cout<<" and "<< PyramidJ[i].size().width<<"x"<<PyramidJ[i].size().height<<endl;
	}

	cout<<"Pyramids created...";
	// init global guesses vg Ag
	Mat vg = Mat::zeros(2,1, CV_64FC1);
	Mat Ag = Mat::eye(2,2,CV_64FC1);

	Mat A(Ag); Mat v(vg);
	// init local guesses v A (for the least precise pyramid level)
	cout<<"Init v = "<< v<<endl<<"A = "<< A <<endl;

	// setting default window for top level of pyramid
	w.setDefaultMask(window);

	for(int L = NumOfLevels-1; L>=0; L--)
	{
		cout<<"----- Level "<<L<<" -----"<<endl;
		//cout<<PyramidI[L]<<endl<<PyramidJ[L]<<endl;
		Image I(PyramidI[L]);
		Image J(PyramidJ[L]);

		// window size can be adjusted if necessary, whole image by default (-1 for purpose of gradient calculation)
		int wx = window.width >> L, wy = window.height >> L;

		// set an upper-left corner of the window
		I.Centralise(Vec2d(int(offset.x) >> L, int(offset.y) >> L));
		J.Centralise(Vec2d(int(offset.x) >> L, int(offset.y) >> L));

		//setting mask
		//w.setMask(mask, wx,wy);
		w.setDefaultMask(Size(wx,wy));

		cout<<"Calculating G (grad matrix)..."<<endl;
		Mat G = calcGradMatrix(I, wx, wy);
		// compute G - gradient matrix
		cout<<"G = "<<endl<<G<<endl;

		//guess from previous level of pyramid
		v = 2*v;
		cout<<"Guess for v = "<<v<<endl;

		int iter = 0;
		while(true)
		{

			J.setTransform(A, v);

			//check if squarediff already less than eps
			double sqdiff = calcSquareDifference(I,J,wx,wy);
			cout<<"--- SQUARE DIFFERENCE = "<<sqdiff<<endl;

			Mat b = calcDiffVector(I, J, wx, wy);
			//cout<<"Diff vector = "<<b<<endl;

			Mat v_opt = G.inv(DECOMP_SVD)*b;
			//cout<<"SVD solution = "<<v_opt<<endl;

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
			cout<<"Shift vector difference = "<<shift<<endl;
			if(shift < eps)
				break;

		}

		affTransform.at<double>(0,0) = A.at<double>(0,0); affTransform.at<double>(0,1) = A.at<double>(0,1);
		affTransform.at<double>(1,0) = A.at<double>(1,0); affTransform.at<double>(1,1) = A.at<double>(1,1);
		affTransform.at<double>(0,2) = v.at<double>(0,0)*(1<<L); affTransform.at<double>(1,2) = v.at<double>(1,0)*(1<<L);

		//if(L>0)
		//{
		//	Mat windowMask = Mat( markPatternAndBackground(PyramidI[L-1], PyramidJ[L-1], 
		//		affTransform, Point2d(int(offset.x) >> L-1, int(offset.y) >> L-1), 
		//		Size(window.width >> L-1, window.height >> L-1)));

		//	imwrite("pat1.jpeg", windowMask); 

		//	w.setMask(windowMask);
		//}
	}	// end of pyramid loop


}

Mat ImageRegistrationLK::markPatternAndBackground(Mat fixed, Mat moving, Mat aff, Point2d pos, Size window)
{
	//Mat fmap(fixed.rows, fixed.cols, fixed.type);

	//Mat trans;
	//warpAffine(fixed, trans, aff, moving.size());

	//Mat fmap = abs(moving - trans);

	Mat A = aff(Range(0,2), Range(0,2));
	Mat T = aff(Range(0,2), Range(2,3));

	Image I(fixed);
	Image J(moving);
	I.Centralise(Vec2d(pos.x, pos.y));
	J.Centralise(Vec2d(pos.x, pos.y));	

	Mat fmap = Mat::ones(window, CV_32FC1);

	double totalMean = 0;
	for(int i=0; i<window.width; i++)
		for(int j=0; j<window.height; j++)
		{
			Mat pt0 = (Mat_<double>(2,1) << i, j);
			Mat pt1 = A*pt0 + T;
			double x = pt1.at<double>(0,0), y = pt1.at<double>(1,0);
			totalMean += (I.at(i,j) - J.at(x,y))*(I.at(i,j) - J.at(x,y));
		}

		totalMean = totalMean/(window.width*window.height);
		fmap = fmap*totalMean;

		for(int i=0; i<window.width-2; i+=3)
			for(int j=0; j<window.height-2; j+=3)
			{
				double mean3x3 = 0;
				for(int x = 0; x<3; x++)
					for(int y=0; y<3; y++)
					{
						Mat pt0 = (Mat_<double>(2,1) << i+x, j+x);
						Mat pt1 = A*pt0 + T;
						double x1 = pt1.at<double>(0,0), y1 = pt1.at<double>(1,0);
						mean3x3 += (I.at(i+x,j+x) - J.at(x1,y1))*(I.at(i+x,j+x) - J.at(x1,y1));
					}
					mean3x3 = mean3x3/9;
					for(int x = 0; x<3; x++)
						for(int y=0; y<3; y++)
						{
							fmap.at<float>(j+y, i+x) = mean3x3;
						}
			}

		return fmap;
}

void ImageRegistrationLK::setMask(Mat m)
{
	m.copyTo(mask);
}

double ImageRegistrationLK::calcSquareDifference(Image I, Image J, int wx, int wy)
{
	double diff = 0;
	for(int x = 0; x < wx; x++)
	{
		for(int y = 0; y < wy; y++)
		{
			double delta = (I.at(x,y) - J.at(x,y));
			diff +=	delta*delta*w.getValue(x,y);
		}
	}
	return diff;
}