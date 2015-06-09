#include "ImageRegistrationLK.h"
#include "MRFSegmentation.h"

void ImageRegistrationLK::WindowFunction::setMask(Mat m, int wx , int wy) 
{ 
	Mat temp;
	// resize to fit appropriate pyramid level
	resize(m, m, Size(0,0), (double)wx/m.size().width, (double)wy/m.size().height, CV_INTER_LINEAR);
	m.convertTo(m, CV_32FC1);
	threshold(m, temp, 0.5, 255, THRESH_BINARY_INV);
	temp.convertTo(temp, CV_8UC1);
	distanceTransform(temp, temp, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	temp.convertTo(windowMask, CV_64FC1);

	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;
	minMaxLoc( windowMask, &minVal, &maxVal, &minLoc, &maxLoc );
	Rmax = maxVal;
	if(Rmax < 0.1)
	{
		setDefaultMask(Size(wx,wy));
		return;
	}


	Mat nim;
	normalize(windowMask, nim,0.0, 255.0, NORM_MINMAX);
	imwrite("m1.jpeg", nim);
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

	Mat nim;
	normalize(windowMask, nim,0.0, 255.0, NORM_MINMAX);
	imwrite("m1.jpeg", nim);
}

double ImageRegistrationLK::WindowFunction::getValue(int x, int y) 
{ 
	//return 1.0;
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
			Mat D2 = (Mat_<double>(6,1) <<  Ix, Iy, x*Ix, y*Ix, x*Iy, y*Iy)*delta;

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
	_isMaskSet=false;
	regMode = 32;
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
	

	for(int i=1; i<NumOfLevels; i++)
	{
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

	//bool lastTwice = false;	// attempt to run last level twice (with old and new mask)

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
		if(_isMaskSet)
			w.setMask(mask, wx,wy);
		else
			w.setDefaultMask(Size(wx,wy));

		cout<<"Calculating G (grad matrix)..."<<endl;
		Mat G = calcGradMatrix(I, wx, wy);
		// compute G - gradient matrix
		cout<<"G = "<<endl<<G<<endl;

		// define parameters
		Mat H0 = getParametrisationMatrix();
		G = H0.t()*G*H0;

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
			if(sqdiff < 0.001)
				break;

			Mat b = calcDiffVector(I, J, wx, wy);
			//cout<<"Diff vector = "<<b<<endl;

			b = H0.t()*b;

			Mat v_opt = G.inv(DECOMP_SVD)*b;
			cout<<"SVD solution = "<<v_opt<<endl;

			Mat tA = interpretResult_affine(v_opt);
			Mat tv = interpretResult_shift(v_opt);

			v = v + A*tv;
			A = A*tA;
			//v = tA*v + tv;
			//A = tA*A;

			cout<<"v = "<<v<<endl;
			cout<<"A = "<<A<<endl;

			iter++;
			if(iter>= MaxIter)
				break;
			double shift = sqrt(tv.dot(tv));
			cout<<"Shift vector difference = "<<shift<<endl;
			if((shift < eps) && (regMode & (REG_SHIFT | REG_AFFINE)))
				break;
			else
			{
				double normA = norm(tA, Mat::eye(tA.size(),tA.type()));
				cout<<"Matrix diff norm = "<<normA<<endl;
				if(normA < eps)
					break;
			}


		}

		affTransform.at<double>(0,0) = A.at<double>(0,0); affTransform.at<double>(0,1) = A.at<double>(0,1);
		affTransform.at<double>(1,0) = A.at<double>(1,0); affTransform.at<double>(1,1) = A.at<double>(1,1);
		affTransform.at<double>(0,2) = v.at<double>(0,0)*(1<<L); affTransform.at<double>(1,2) = v.at<double>(1,0)*(1<<L);

		if(L>0)
		{
			Mat windowMask = Mat( markPatternAndBackground(PyramidI[L-1], PyramidJ[L-1], 
				affTransform, Point2d(int(offset.x) >> L-1, int(offset.y) >> L-1), 
				Size(window.width >> L-1, window.height >> L-1)));

			imwrite("afterThreshold.jpg", windowMask);
			// segmentation with 2 labels
			windowMask = patternSegmentation(windowMask, 2);

			imwrite("afterMRF.jpg", windowMask);

			windowMask.copyTo(mask);
			_isMaskSet = true;
		}
		//else
		//{
		//	if(!lastTwice)
		//	{
		//		// trying to run last level of pyramid twice for better precision
		//		Mat windowMask = Mat( markPatternAndBackground(PyramidI[0], PyramidJ[0], 
		//		affTransform, Point2d(int(offset.x), int(offset.y)), 
		//		Size(window.width, window.height)));
		//		L++;

		//		imwrite("afterThreshold.jpg", windowMask);
		//		// segmentation with 2 labels
		//		windowMask = patternSegmentation(windowMask, 2);

		//		imwrite("afterMRF.jpg", windowMask);

		//		windowMask.copyTo(mask);
		//		_isMaskSet = true;

		//		lastTwice = true;
		//	}
		//}
	}	// end of pyramid loop


}

Mat ImageRegistrationLK::markPatternAndBackground(Mat fixed, Mat moving, Mat aff, Point2d pos, Size window)
{
	Mat A = aff(Range(0,2), Range(0,2));
	Mat T = aff(Range(0,2), Range(2,3));

	Image I(fixed);
	Image J(moving);
	I.Centralise(Vec2d(pos.x, pos.y));
	J.Centralise(Vec2d(pos.x, pos.y));

	J.setTransform(A,T);

	Mat fmap = Mat::ones(window, CV_32FC1);

	double totalMean = 0;
	for(int i=0; i<window.width; i++)
		for(int j=0; j<window.height; j++)
		{
			//Mat pt0 = (Mat_<double>(2,1) << i, j);
			//Mat pt1 = A*pt0 + T;
			//double x = pt1.at<double>(0,0), y = pt1.at<double>(1,0);
			double diff = I.at(i,j) - J.at(i,j);
			totalMean += diff*diff;
		}

		totalMean = totalMean/(window.width*window.height);

		for(int i=0; i<window.width-2; i+=3)
			for(int j=0; j<window.height-2; j+=3)
			{
				double mean3x3 = 0;
				for(int x = 0; x<3; x++)
					for(int y=0; y<3; y++)
					{
						//Mat pt0 = (Mat_<double>(2,1) << i+x, j+y);
						//Mat pt1 = A*pt0 + T;
						//double x1 = pt1.at<double>(0,0), y1 = pt1.at<double>(1,0);
						double diff = I.at(i+x,j+y) - J.at(i+x,j+y);
						mean3x3 += diff*diff;
					}
					mean3x3 = mean3x3/9;
					for(int x = 0; x<3; x++)
						for(int y=0; y<3; y++)
						{
							fmap.at<float>(j+y, i+x) = float(mean3x3);
						}
			}
		normalize(fmap, fmap, 0.0, 255.0, NORM_MINMAX);
		return fmap;
}

void ImageRegistrationLK::setMask(Mat m)
{
	m.copyTo(mask);
	_isMaskSet = true;
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

Mat ImageRegistrationLK::getDifference()
{
	//Mat invA;// = (Mat_<double>(2,3) << 1.0, 0.0, -5.0, 0.0, 1.0, 5.0);
	//invertAffineTransform(affTransform, invA);
	//Mat shift = (Mat(2,1,CV_64FC1) << offset.x, offset.y);
	//shift = invA*shift;
	Image I(fixed);
	I.Centralise(offset);
	//Mat J1;
	//warpAffine(moving, J1, invA, fixed.size());
	Image J(moving);
	J.Centralise(offset);
	Mat A = affTransform(Range(0,2), Range(0,2));
	Mat d =   affTransform(Range(0,2), Range(2,3));
	J.setTransform(A,d);
	Mat diff(window, CV_32FC1);
	for(int i=0; i<window.width;i++)
		for(int j=0; j<window.height; j++)
		{
			diff.at<float>(j,i) = abs(I.at(i, j) - J.at(i, j));//*w.getValue(i,j);
		}
	return diff;
}

void ImageRegistrationLK::setRegistrationMode(int _regMode)
{
	regMode = _regMode;
}

Mat ImageRegistrationLK::getParametrisationMatrix()
{
	// Order of params: dx, dy, \theta, s(scew), a, b (scale for x and y)
	//
	// If Affine, return unary matrix

	if(regMode & REG_AFFINE)
		return Mat::eye(Size(6,6),CV_64FC1);

	int cols = 0;
	if(regMode & REG_SHIFT)
		cols += 2;
	if(regMode & REG_ROTATION)
		cols += 1;
	if(regMode & REG_SCALE)
		cols += 2;
	else if(regMode & REG_UNIFORM_SCALE)
		cols += 1;
	if(regMode & REG_SCEW)
		cols += 1;

	Mat H0(6, cols, CV_64FC1);
	int k=0;
	if(regMode & REG_SHIFT) 
	{
		H0.at<double>(0,k) = 1; H0.at<double>(1,k) = 0; H0.at<double>(2,k) = 0; H0.at<double>(3,k) = 0; H0.at<double>(4,k) = 0; H0.at<double>(5,k) = 0;
		k++;
		H0.at<double>(0,k) = 0; H0.at<double>(1,k) = 1; H0.at<double>(2,k) = 0; H0.at<double>(3,k) = 0; H0.at<double>(4,k) = 0; H0.at<double>(5,k) = 0;
		k++;
	}
	if(regMode & REG_ROTATION)
	{
		H0.at<double>(0,k) = 0; H0.at<double>(1,k) = 0; H0.at<double>(2,k) = 0; H0.at<double>(3,k) = -1; H0.at<double>(4,k) = 1; H0.at<double>(5,k) = 0;
		k++;
	}
	if(regMode & REG_SCEW)
	{
		H0.at<double>(0,k) = 0; H0.at<double>(1,k) = 0; H0.at<double>(2,k) = 0; H0.at<double>(3,k) = 1; H0.at<double>(4,k) = 0; H0.at<double>(5,k) = 0;
		k++;
	}
	if(regMode & REG_SCALE)
	{
		H0.at<double>(0,k) = 0; H0.at<double>(1,k) = 0; H0.at<double>(2,k) = 1; H0.at<double>(3,k) = 0; H0.at<double>(4,k) = 0; H0.at<double>(5,k) = 0;
		k++;
		H0.at<double>(0,k) = 0; H0.at<double>(1,k) = 0; H0.at<double>(2,k) = 0; H0.at<double>(3,k) = 0; H0.at<double>(4,k) = 0; H0.at<double>(5,k) = 1;
		k++;
	}
	else if(regMode & REG_UNIFORM_SCALE)
	{
		H0.at<double>(0,k) = 0; H0.at<double>(1,k) = 0; H0.at<double>(2,k) = 1; H0.at<double>(3,k) = 0; H0.at<double>(4,k) = 0; H0.at<double>(5,k) = 1;
		k++;
	}

	return H0;
}
Mat ImageRegistrationLK::interpretResult_shift(Mat v_opt)
{
	Mat tv(2,1,CV_64FC1);
	if((regMode & REG_SHIFT) || (regMode & REG_AFFINE))
	{
		tv.at<double>(0,0) = v_opt.at<double>(0,0);
		tv.at<double>(1,0) = v_opt.at<double>(1,0);
	}
	else
	{
		tv.at<double>(0,0) = 0;
		tv.at<double>(1,0) = 0;
	}
	return tv;
}
Mat ImageRegistrationLK::interpretResult_affine(Mat v_opt)
{
	if(regMode & REG_AFFINE)
	{
		Mat tA(2,2,CV_64FC1);
		tA.at<double>(0,0) = 1+v_opt.at<double>(2,0); tA.at<double>(0,1) = v_opt.at<double>(3,0);
		tA.at<double>(1,0) = v_opt.at<double>(4,0); tA.at<double>(1,1) = 1+v_opt.at<double>(5,0);
		return tA;
	}

	int k=0;
	if(regMode & REG_SHIFT)
		k += 2;
	Mat rot = Mat::eye(2,2,CV_64FC1);
	if(regMode & REG_ROTATION)
	{
		double theta = v_opt.at<double>(k,0);
		k++;
		rot.at<double>(0,0) = cos(theta); rot.at<double>(0,1) = -sin(theta);
		rot.at<double>(1,0) = sin(theta); rot.at<double>(1,1) = cos(theta);
	}
	Mat scew = Mat::eye(2,2,CV_64FC1);
	if(regMode & REG_SCEW)
	{
		rot.at<double>(0,1) = v_opt.at<double>(k,0);
		k++;
	}
	Mat scale = Mat::eye(2,2,CV_64FC1);
	if(regMode & REG_SCALE)
	{
		double a = v_opt.at<double>(k,0);
		k++;
		double b = v_opt.at<double>(k,0);
		scale.at<double>(0,0) = 1+a; scale.at<double>(0,1) = 0;
		scale.at<double>(1,0) = 0; scale.at<double>(1,1) = 1+b;
	}
	else if(regMode & REG_UNIFORM_SCALE)
	{
		double a = v_opt.at<double>(k,0);
		scale.at<double>(0,0) = 1+a; scale.at<double>(0,1) = 0;
		scale.at<double>(1,0) = 0; scale.at<double>(1,1) = 1+a;
	}

	return rot*scew*scale; 
}