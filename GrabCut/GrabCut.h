#pragma once
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/detail/gcgraph.hpp"
#include "GMM.h"

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
    
private:
    double calcBeta(const Mat& img);
    void calcSmoothTerm(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma);
    void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM);
    void assignGMMsComponents(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs);
    void learnGMMs(const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM);
    void constructGCGraph(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda, const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW, detail::GCGraph<double>& graph);
    void estimateSegmentation(detail::GCGraph<double>& graph, Mat& mask);
};
