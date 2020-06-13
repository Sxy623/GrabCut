#include "BorderMatting.h"

// 构造函数
BorderMatting::BorderMatting() {
    sections = 0;
    pointOnContourCount = 0;
}

// 边缘检测
void BorderMatting::borderDetect(const Mat& mask) {
    edge = mask & 1;
    edge.convertTo(edge, CV_8UC1, 255);
    Mat tmpEdge;
    Canny(edge, tmpEdge, 3, 9);
    tmpEdge.convertTo(tmpEdge, CV_8UC1);
    edge = tmpEdge;
}

// 用DFS找到所有轮廓点
void BorderMatting::paraContour() {
    Mat flagEdge;
    flagEdge.create(edge.size(), CV_8UC1);
    flagEdge.setTo(Scalar(0));
    for (int y = 0; y < edge.rows; y++) {
        for (int x = 0; x < edge.cols; x++) {
            if (edge.at<uchar>(y, x) != 0 && flagEdge.at<uchar>(y, x) == 0) {
                sections++;
                dfs(x, y, flagEdge);
            }
        }
    }
}

// 深度优先搜索
void BorderMatting::dfs(int x, int y, Mat& flagEdge) {
    const int findContourStep = 8;
    const int contourStepX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int contourStepY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    flagEdge.at<uchar>(y, x) = 255;
    PointOnContour pt;
    pt.p.x = x;
    pt.p.y = y;
    pt.section = sections;
    pt.index = pointOnContourCount;
    ++pointOnContourCount;
    contour.push_back(pt);
    for (int i = 0; i < findContourStep; i++) {
        int stepX = contourStepX[i];
        int stepY = contourStepY[i];
        int newX = x + stepX;
        int newY = y + stepY;
        // 判断是否超过边界
        if (newX < 0 || newX > edge.cols - 1 || newY < 0 || newY > edge.rows - 1) {
            continue;
        }
        // 判断是否是未标记的轮廓
        if (edge.at<uchar>(newY, newX) == 0 || flagEdge.at<uchar>(newY, newX) != 0) {
            continue;
        }
        dfs(newX, newY, flagEdge);
    }
}

// 构建条带
void BorderMatting::buildStrip(const Mat& mask) {
    const int findContourStep = 8;
    const int contourStepX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int contourStepY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    Mat flagMask;
    flagMask.create(mask.size(), CV_8UC1);
    flagMask.setTo(Scalar(0));
    vector<Point> tmpPointOnStrip;
    for (int i = 0; i < contour.size(); i++) {
        PointOnStrip pt;
        pt.p = contour[i].p;
        pt.distance = 0;
        pt.area = contour[i].index;
        int key = pt.p.y * mask.cols + pt.p.x;
        strip[key] = pt;
        tmpPointOnStrip.push_back(pt.p);
        flagMask.at<uchar>(pt.p.y, pt.p.x) = 255;
    }
    // 宽度优先搜索
    int num = 0;
    while (num < tmpPointOnStrip.size()) {
        Point pt = tmpPointOnStrip[num];
        ++num;
        int key = pt.y * mask.cols + pt.x;
        PointOnStrip ptOnStrip = strip[key];
        // 判断是否超出宽度
        if (abs(ptOnStrip.distance) >= stripWidth) {
            continue;
        }
        for (int i = 0; i < findContourStep; i++) {
            int stepX = contourStepX[i];
            int stepY = contourStepY[i];
            int newX = ptOnStrip.p.x + stepX;
            int newY = ptOnStrip.p.y + stepY;
            // 判断是否超过边界
            if (newX < 0 || newX > edge.cols - 1 || newY < 0 || newY > edge.rows - 1) {
                continue;
            }
            // 判断是否标记
            if (flagMask.at<uchar>(newY, newX) != 0) {
                continue;
            }
            PointOnStrip newPtOnStrip;
            newPtOnStrip.p.x = newX;
            newPtOnStrip.p.y = newY;
            newPtOnStrip.distance = abs(ptOnStrip.distance) + 1;
            if ((mask.at<uchar>(newY, newX) & 1) == 0) {
                newPtOnStrip.distance = -newPtOnStrip.distance;
            }
            newPtOnStrip.area = ptOnStrip.area;
            tmpPointOnStrip.push_back(newPtOnStrip.p);
            int key = newPtOnStrip.p.y * mask.cols + newPtOnStrip.p.x;
            strip[key] = newPtOnStrip;
            flagMask.at<uchar>(newPtOnStrip.p.y, newPtOnStrip.p.x) = 255;
        }
    }
}

// 能量最小化
void BorderMatting::minimizeEnergy(const Mat& img, const Mat& mask) {
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    grayImg.convertTo(grayImg, CV_32FC1, 1.0 / 255.0);
    for (int i = 0; i < contour.size(); i++) {
        MeanAndVar meanAndCov;
        calMeanAndVar(contour[i].p, grayImg, mask, meanAndCov);
        for (int d0 = 0; d0 < deltaLevels; d0++) {
            for (int s0 = 0; s0 < sigmaLevels; s0++) {
                double delta0 = delta(d0);
                double sigma0 = sigma(s0);
                energyFuncVals[contour[i].index][d0][s0] = INFINITY;
                ParaSigAndDel paraSigAndDel;
                paraSigAndDel.delta = delta0;
                paraSigAndDel.sigma = sigma0;
                double dVal = calDataTerm(contour[i].p, grayImg.at<float>(contour[i].p.y, contour[i].p.x), delta0, sigma0, meanAndCov, 0);
                if (contour[i].index == 0) {
                    energyFuncVals[contour[i].index][d0][s0] = dVal;
                    continue;
                }
                for (int d1 = 0; d1 < deltaLevels; d1++) {
                    for (int s1 = 0; s1 < sigmaLevels; s1++) {
                        double delta1 = delta(d1);
                        double sigma1 = sigma(s1);
                        double vVal = 0;
                        if (contour[i - 1].section == contour[i].section) {
                            ParaSigAndDel tmpParaSigAndDel;
                            tmpParaSigAndDel.delta = delta0 - delta1;
                            tmpParaSigAndDel.sigma = sigma0 - sigma1;
                            vVal = calSmoothTerm(tmpParaSigAndDel);
                        }
                        double result = energyFuncVals[contour[i].index - 1][d1][s1] + dVal + vVal;
                        if (result < energyFuncVals[contour[i].index][d0][s0]) {
                            ParaSigAndDel paraSigAndDel;
                            paraSigAndDel.delta = delta1;
                            paraSigAndDel.sigma = sigma1;
                            energyFuncVals[contour[i].index][d0][s0] = result;
                            paraSigAndDels[contour[i].index][d0][s0] = paraSigAndDel;
                        }
                    }
                }
            }
        }
    }
    double minE = INFINITY;
    ParaSigAndDel minParaSigAndDel;
    paraSigAndDelsVec = vector<ParaSigAndDel>(pointOnContourCount);
    for (int d = 0; d < deltaLevels; d++) {
        for (int s = 0; s < sigmaLevels; s++) {
            if (energyFuncVals[pointOnContourCount - 1][d][s] < minE) {
                minE = energyFuncVals[pointOnContourCount - 1][d][s];
                minParaSigAndDel.delta = d;
                minParaSigAndDel.delta = s;
            }
        }
    }
    
    paraSigAndDelsVec[pointOnContourCount - 1] = minParaSigAndDel;
    for (int i = pointOnContourCount - 2; i >= 0; i--) {
        paraSigAndDelsVec[i] = paraSigAndDels[i + 1][(int)(paraSigAndDelsVec[i + 1].delta)][(int)(paraSigAndDelsVec[i + 1].sigma)];
    }
}

// 采样计算均值和方差
void BorderMatting::calMeanAndVar(Point p, const Mat& img, const Mat& mask, MeanAndVar& meanAndVar) {
    const int halfL = 20;
    double backMean = 0, frontMean = 0;
    double backVariance = 0, frontVariance = 0;
    int frontCounter = 0, backCounter = 0;
    int x = (p.x - halfL < 0) ? 0 : p.x - halfL;
    int width = (x + 2 * halfL + 1 <= img.cols) ? halfL * 2 + 1 : img.cols - x;
    int y = (p.y - halfL < 0) ? 0 : p.y - halfL;
    int height = (y + 2 * halfL + 1 <= img.rows) ? halfL * 2 + 1 : img.rows - y;
    Mat neiborPixels = img(Rect(x, y, width, height));
    for (int i = 0; i < neiborPixels.rows; i++) {
        for (int j = 0; j < neiborPixels.cols; j++) {
            int pixelColor = neiborPixels.at<uchar>(i, j);
            if ((mask.at<uchar>(y + i, x + j) & 1) == 1) {
                frontMean += pixelColor;
                frontCounter++;
            }
            else {
                backMean += pixelColor;
                backCounter++;
            }
        }
    }
    
    if (frontCounter > 0) {
        frontMean = frontMean / frontCounter;
    }
    else {
        frontMean = 0;
    }
    if (backCounter > 0) {
        backMean = backMean / backCounter;
    }
    else {
        backMean = 0;
    }
    
    for (int i = 0; i < neiborPixels.rows; i++) {
        for (int j = 0; j < neiborPixels.cols; j++) {
            int pixelColor = neiborPixels.at<uchar>(i, j);
            if ((mask.at<uchar>(y + i, x + j) & 1) == 1)
                frontVariance += (frontMean - pixelColor) * (frontMean - pixelColor);
            else
                backVariance += (pixelColor - backMean) * (pixelColor - backMean);
        }
    }
    
    if (frontCounter > 0) {
        frontVariance = frontVariance / frontCounter;
    }
    else {
        frontVariance = 0;
    }
    if (backCounter > 0) {
        backVariance = backVariance / backCounter;
    }
    else {
        backVariance = 0;
    }
    
    meanAndVar.backMean = backMean;
    meanAndVar.backVar = backVariance;
    meanAndVar.frontMean = frontMean;
    meanAndVar.frontVar = frontVariance;
}

double BorderMatting::delta(int level) {
    return (double)(2 * stripWidth) / deltaLevels * level - stripWidth;
    
}
double BorderMatting::sigma(int level) {
    return (double)stripWidth / (double)sigmaLevels * (level + 1);
}

double BorderMatting::calDataTerm(Point p, double z, double delta, double sigma, MeanAndVar& meanAndVar, double distance) {
    double alpha = Sigmoid(distance, delta, sigma);
    double MmeanValue = Mmean(alpha, meanAndVar.frontMean, meanAndVar.backMean);
    double MvarValue  = Mvar(alpha, meanAndVar.frontVar, meanAndVar.backVar);
    double D = Gaussian(z, MmeanValue, MvarValue);
    D = -log(D) / log(2.0);
    return D;
}

double BorderMatting::calSmoothTerm(ParaSigAndDel paraSigAndDel) {
    return lambda1 * paraSigAndDel.delta * paraSigAndDel.delta + lambda2 * paraSigAndDel.sigma * paraSigAndDel.sigma;
}

double BorderMatting::Gaussian(double x, double mean, double sigma) {
    double res = 1.0 / (pow(sigma, 0.5)*pow(2.0*PI, 0.5))* exp(-(pow(x - mean, 2.0) / (2.0*sigma)));
    return res;
}

double BorderMatting::Mmean(double alfa, double Fmean, double Bmean) {
    return (1.0 - alfa)*Bmean + alfa*Fmean;
}

double BorderMatting::Mvar(double alfa, double Fvar, double Bvar) {
    return (1.0 - alfa)*(1.0 - alfa)*Bvar + alfa*alfa*Fvar;
}

double BorderMatting::Sigmoid(double dis, double deltaCenter, double sigma) {
    if (dis < deltaCenter - sigma / 2)
        return 0;
    if (dis >= deltaCenter + sigma / 2)
        return 1;
    double res = -(dis - deltaCenter) / sigma;
    res = exp(res);
    res = 1.0 / (1.0 + res);
    return res;
}

void BorderMatting::calMaskAlpha(const Mat& mask, Mat& alphaMask) {
    const int findContourStep = 8;
    const int contourStepX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int contourStepY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    Mat mask2;
    mask2.create(mask.size(), CV_8UC1);
    
    alphaMask.create(mask.size(), CV_32FC1);
    alphaMask.setTo(Scalar(0));
    Mat flagMask;
    flagMask.create(mask.size(), CV_8UC1);
    flagMask.setTo(Scalar(0));
    vector<PointOnStrip> tmpPointOnStrip;
    for (int i = 0; i < contour.size(); i++) {
        PointOnStrip pt;
        pt.p = contour[i].p;
        pt.distance = 0;
        pt.area = contour[i].index;
        tmpPointOnStrip.push_back(pt);
        flagMask.at<uchar>(pt.p.y, pt.p.x) = 255;
        ParaSigAndDel tmpSigAndDel;
        tmpSigAndDel = paraSigAndDelsVec[pt.area];
        tmpSigAndDel.delta = delta(tmpSigAndDel.delta);
        tmpSigAndDel.sigma = sigma(tmpSigAndDel.sigma);
        double alpha = Sigmoid(pt.distance, 0, tmpSigAndDel.sigma);
        alphaMask.at<float>(pt.p.y, pt.p.x) = alpha;
    }
    // 宽度优先搜索
    int num = 0;
    while (num < tmpPointOnStrip.size()) {
        PointOnStrip pt = tmpPointOnStrip[num];
        ++num;
        for (int i = 0; i < findContourStep; i++) {
            int stepX = contourStepX[i];
            int stepY = contourStepY[i];
            int newX = pt.p.x + stepX;
            int newY = pt.p.y + stepY;
            // 判断是否超过边界
            if (newX < 0 || newX > edge.cols - 1 || newY < 0 || newY > edge.rows - 1) {
                continue;
            }
            // 判断是否标记
            if (flagMask.at<uchar>(newY, newX) != 0) {
                continue;
            }
            PointOnStrip newPtOnStrip;
            newPtOnStrip.p.x = newX;
            newPtOnStrip.p.y = newY;
            newPtOnStrip.distance = abs(pt.distance) + 1;
            newPtOnStrip.area = pt.area;
            if ((mask.at<uchar>(newY, newX) & 1) == 0) {
                newPtOnStrip.distance = -newPtOnStrip.distance;
            }
            tmpPointOnStrip.push_back(newPtOnStrip);
            flagMask.at<uchar>(newPtOnStrip.p.y, newPtOnStrip.p.x) = 255;
            ParaSigAndDel tmpSigAndDel;
            tmpSigAndDel = paraSigAndDelsVec[newPtOnStrip.area];
            tmpSigAndDel.delta = delta(tmpSigAndDel.delta);
            tmpSigAndDel.sigma = sigma(tmpSigAndDel.sigma);
            double alpha = Sigmoid(newPtOnStrip.distance, 0, tmpSigAndDel.sigma);
            alphaMask.at<float>(newY, newX) = alpha;
            mask2.at<uchar>(newY, newX) = uchar(alpha * 255);
        }
    }
    imshow("mask", mask2);
}

void BorderMatting::display(const Mat& img, const Mat& alphaMask) {
    vector<Mat> imgVec(3);
    vector<Mat> bgVec(3);
    Mat newImg;
    img.convertTo(newImg, CV_32FC3, 1.0 / 255.0);
    split(newImg, imgVec);
    Mat bg = Mat(img.size(), CV_32FC3, Scalar(0, 0, 0));
    split(bg, bgVec);
    imgVec[0] = imgVec[0].mul(alphaMask) + bgVec[0].mul(1.0 - alphaMask);
    imgVec[1] = imgVec[1].mul(alphaMask) + bgVec[1].mul(1.0 - alphaMask);
    imgVec[2] = imgVec[2].mul(alphaMask) + bgVec[2].mul(1.0 - alphaMask);
    Mat result;
    merge(imgVec, result);
    imshow("boader matting", result);
    waitKey(0);
}

void BorderMatting::run(const Mat& img, const Mat& mask) {
    borderDetect(mask);
    paraContour();
    buildStrip(mask);
    minimizeEnergy(img, mask);
    Mat alphaMask;
    calMaskAlpha(mask, alphaMask);
    display(img, alphaMask);
}
