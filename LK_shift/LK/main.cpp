
#include "utils.hxx"



int main(int argc, char *argv[])
{
	int wx = 5, wy = 5;
	double dx = 0.3, dy = 0.5;

	Mat fixed(wy, wx, CV_64FC1);
	for(int i=0; i<wx; i++)
	{
		for(int j=0; j<wy; j++)
		{
			if((i==0) || (i == wx-1) || (j ==0) || (j == wy-1))
				fixed.at<double>(j, i) = 0;
			else
				fixed.at<double>(j, i) = (i+j)*40.0;
		}
	}
	cout<<fixed<<endl;
	Mat moved(wy,wx,CV_64FC1);
	for(int i=0; i<wx; i++)
	{
		for(int j=0; j<wy; j++)
		{
			double x1 = floor(i+dx), x2 = ceil(i+dx), y1 = floor(j+dy), y2 = ceil(j+dy);
			bool bx1 = (x1<0) || (x1>=wx), bx2 = (x2<0) || (x2>=wx), by1 = (y1<0) || (y1>=wy), by2 = (y2<0) || (y2>=wy);
			double p11 = (bx1 || by1)? 0:  fixed.at<double>(y1, x1);
			double p12 = (bx1 || by2)? 0:  fixed.at<double>(y2, x1);
			double p21 = (bx2 || by1)? 0:  fixed.at<double>(y1, x2);
			double p22 = (bx2 || by2)? 0:  fixed.at<double>(y2, x2);
			double r1 = dx*p21 + (1-dx)*p11, r2 = dx*p22 + (1-dx)*p12;
			double r = dy*r2 + (1-dy)*r1;
			moved.at<double>(j,i) =r;
		}
	}
	cout<<moved<<endl;

	int MaxIter = 25;	// could be set by user
	double eps = 0.001;	// could be set by user

	// init global guesses vg Ag
	Mat vg = Mat::zeros(2,1, CV_64FC1);
	Mat Ag = Mat::eye(2,2,CV_64FC1);

	
	Mat A(Ag); Mat v(vg);
	// init local guesses v A (for the least precise pyramid level)
	cout<<"Init v = "<< v<<endl;

	for(int L = 0; L>=0; L--)
	{
		cout<<"----- Level "<<L<<" -----"<<endl;
		Image I(fixed);
		Image J(moved);

		// window size can be adjusted if necessary, whole image by default (-1 for purpose of gradient calculation)
		//int wx = PyramidI[L].size().width/2-1, wy = PyramidI[L].size().height/2-1;

		//translate image I (to do it centralized at the point of interest -- may be as in ITK centralize to mass center)
		//Moments mI = moments(PyramidI[L]);
		//Vec2d uI( int(mI.m10/mI.m00), int(mI.m01/mI.m00));		// -- mass center to centralize fixed image, also a point of interest, since there's no
		//										// feature point for entire picture or some region
		//Vec2d uI(PyramidI[L].size().width/2-1, PyramidI[L].size().height/2-1);
		//I.Centralise(uI);
		//cout<<"Center for I = "<< uI<<endl;

		////Moments mJ = moments(PyramidJ[L]);
		////Vec2d uJ( int(mJ.m10/mJ.m00), int(mJ.m01/mJ.m00));
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

			cout<<G*v_opt - b<<endl;

			v.at<double>(0,0) += v_opt.at<double>(0,0);
			v.at<double>(1,0) += v_opt.at<double>(1,0);
			//A.at<double>(0,0) = 1+v_opt.at<double>(2,0); A.at<double>(0,1) = v_opt.at<double>(3,0);
			//A.at<double>(1,0) = v_opt.at<double>(4,0); A.at<double>(1,1) = 1+v_opt.at<double>(5,0);

			cout<<"v = "<<v<<endl;
			cout<<"A = "<<endl<<A<<endl;

			iter++;
			if(iter>= MaxIter)
				break;
			double shift = sqrt(v_opt.dot(v_opt));
			cout<<"Shift = "<<shift<<endl;
			if(shift < eps)
				break;
			
		}

	}	// end of pyramid loop

	Mat aff(2,3,CV_64FC1);
	aff.at<double>(0,0) = A.at<double>(0,0); aff.at<double>(0,1) = A.at<double>(0,1);
	aff.at<double>(1,0) = A.at<double>(1,0); aff.at<double>(1,1) = A.at<double>(1,1);
	aff.at<double>(0,2) = v.at<double>(0,0); aff.at<double>(1,2) = v.at<double>(1,0);

	//Mat newimg;
	//warpAffine(moving, newimg, aff, moving.size());

	//imwrite("new.jpeg", newimg);

}
