#pragma once
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

enum {
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};

class GrabCut2D {
public:
    void GrabCut(InputArray _img, InputOutputArray _mask, Rect rect, InputOutputArray _bgdModel, InputOutputArray _fgdModel, int iterCount, int mode);

	~GrabCut2D(void);
};

