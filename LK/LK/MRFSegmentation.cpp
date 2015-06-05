#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MRFSegmentation.h"

Mat patternSegmentation(Mat I, int num_labels)
{
	Mat res(I.size(), I.type());
	int num_pixels = I.cols*I.rows, width = I.size().width, height = I.size().height;

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (I.at<float>(i/width, i%width) < 0.5 )
			{
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 1;
			}
			else 
			{
				if(  l == 0 ) data[i*num_labels+l] = 10;
				else data[i*num_labels+l] = 0;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 400  ? (l1-l2)*(l1-l2):400;


	try
	{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(3);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			res.at<float>(i/I.cols, i%I.cols) = gc->whatLabel(i)*255;

		delete gc;

		//imwrite("after-mrf.jpeg", res);
	}
	catch (GCException e)
	{
		e.Report();
	}

	delete [] smooth;
	delete [] data;

	return res;
}

