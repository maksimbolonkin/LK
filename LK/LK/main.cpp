
#include "utils.hxx"



int main(int argc, char *argv[])
{

	if(argc<3)
	{
		cout<<"Enter two filenames: fixed image and moving image."<<endl;
		exit(-1);
	}

	int MaxIter = 25;	// could be set by user
	double eps = 0.001;	// could be set by user

	Mat fixed = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	fixed.convertTo(fixed, CV_64FC1);
	Mat moving = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	moving.convertTo(moving, CV_64FC1);
	// read the images I J

	cout<<"Images have been read..."<<endl;

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

	Mat newimg;
	warpAffine(fixed, newimg, aff, moving.size());

	imwrite("new.jpeg", newimg);

}
