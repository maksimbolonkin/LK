#pragma once

#include <string.h>
#include <time.h>
#include <opencv\cv.h>
#include <iostream>
#include <opencv\highgui.h>
#include <math.h>
#include <calib3d\calib3d.hpp>
#include "GCO\GCoptimization.h"

using namespace std;
using namespace cv;

Mat patternSegmentation(Mat I, int num_labels);