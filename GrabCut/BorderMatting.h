#pragma once
#include <iostream>
#include <map>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

const double PI = 3.14159;

struct PointOnContour {
    Point p;
    int section;
    int index;
};

struct PointOnStrip {
    Point p;
    double distance;
    int area;
};

struct MeanAndVar {
    double backMean;
    double frontMean;
    double backVar;
    double frontVar;
};

struct ParaSigAndDel {
    double sigma;
    double delta;
};

class BorderMatting {
public:
    BorderMatting();
    void borderDetect(const Mat& mask);
    void paraContour();
    void buildStrip(const Mat& mask);
    void minimizeEnergy(const Mat& img, const Mat& mask);
    void calMaskAlpha(const Mat& mask, Mat& alphaMask);
    void display(const Mat& img, const Mat& alphaMask);
    void run(const Mat& img, const Mat& mask);
private:
    void dfs(int x, int y, Mat& flagEdge);
    void calMeanAndVar(Point p, const Mat& img, const Mat& mask, MeanAndVar& meanAndVar);
    double delta(int level);
    double sigma(int level);
    double calDataTerm(Point p, double z, double delta, double sigma, MeanAndVar& meanAndVar, double distance);
    double calSmoothTerm(ParaSigAndDel paraSigAndDel);
    double Gaussian(double x, double mean, double sigma);
    double Mmean(double x, double Fmean, double Bmean);
    double Mvar(double x, double Fvar, double Bvar);
    double Sigmoid(double dis, double deltaCenter, double sigma);
    
    Mat edge;
    vector<PointOnContour> contour;
    int sections;
    int pointOnContourCount;
    map<int, PointOnStrip> strip;
    const int stripWidth = 6;
    
    static const int deltaLevels = 30;
    static const int sigmaLevels = 10;
    static const int contourSize = 1000000;
    static const int lambda1 = 50;
    static const int lambda2 = 1000;
    double energyFuncVals[contourSize][deltaLevels][sigmaLevels];
    ParaSigAndDel paraSigAndDels[contourSize][deltaLevels][sigmaLevels];
    vector<ParaSigAndDel> paraSigAndDelsVec;
};
