#pragma once

#include <string.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <opencv2/calib3d.hpp>
#include "GCO/GCoptimization.h"

using namespace std;
using namespace cv;

Mat patternSegmentation(Mat I, int num_labels);
