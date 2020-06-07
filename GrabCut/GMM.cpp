#include "GMM.h"

// 构造函数，图片的背景和前景分别对应一个高斯混合模型
GMM::GMM(Mat& _model) {
    // 高斯分布个数：componentsCount
    // 每个高斯分布的参数个数：modelSize
    const int modelSize = 3 + 9 + 1;  // 3个均值，3*3个协方差，1个权值
    if (_model.empty()) {
        // 初始化参数矩阵
        _model.create(1, modelSize * componentsCount, CV_64FC1);
        _model.setTo(Scalar(0));
    }
    else if ((_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize * componentsCount))
        CV_Error(Error::StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");
    
    model = _model;
    
    // 存储方式：componentsCount个coefs + 3 * componentsCount个mean + 3 * 3 * componentsCount个cov
    coefs = model.ptr<double>(0);    // 高斯分布权值变量起始存储指针
    mean = coefs + componentsCount;  // 均值变量起始存储指针
    cov = mean + 3*componentsCount;  // 协方差变量起始存储指针
    
    // 计算各个高斯分布的协方差矩阵的逆和行列式
    for (int ci = 0; ci < componentsCount; ci++)
        if (coefs[ci] > 0)
            calcInverseCovAndDeterm(ci);
}
  
// 计算单个像素（由color=（B,G,R）三维double型向量来表示）属于这个高斯混合模型的概率
double GMM::operator()(const Vec3d color) const {
    double res = 0;
    for (int ci = 0; ci < componentsCount; ci++)
        res += coefs[ci] * (*this)(ci, color);
    return res;
}
  
// 计算单个像素（由color=（B,G,R）三维double型向量来表示）在第ci个高斯分布上的概率
double GMM::operator()(int ci, const Vec3d color) const {
    double res = 0;
    if (coefs[ci] > 0) {
        CV_Assert(covDeterms[ci] > std::numeric_limits<double>::epsilon());
        Vec3d diff = color;
        double* m = mean + 3 * ci;
        diff[0] -= m[0];
        diff[1] -= m[1];
        diff[2] -= m[2];
        double mult = diff[0] * (diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
            + diff[1] * (diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
            + diff[2] * (diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
    }
    return res;
}
  
// 返回单个像素最有可能属于的高斯分布（即概率最大的那个）
int GMM::whichComponent(const Vec3d color) const {
    int k = 0;
    double max = 0;
  
    for (int ci = 0; ci < componentsCount; ci++) {
        double p = (*this)(ci, color);
        if (p > max) {
            k = ci;
            max = p;
        }
    }
    return k;
}
  
// 参数学习前的初始化
void GMM::initLearning() {
    for (int ci = 0; ci < componentsCount; ci++) {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}
  
// 增加样本
void GMM::addSample(int ci, const Vec3d color) {
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}
  
// 计算每一个高斯分布的权值、均值和协方差矩阵
void GMM::endLearning() {
    const double variance = 0.01;
    for (int ci = 0; ci < componentsCount; ci++) {
        int n = sampleCounts[ci];
        if (n == 0) coefs[ci] = 0;
        else {
            // 权值系数
            coefs[ci] = (double)n/totalSampleCount;
  
            // 均值
            double* m = mean + 3 * ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;
  
            // 协方差
            double* c = cov + 9 * ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];
  
            // 如果行列式小于等于0，为对角线元素增加白噪声，避免退化
            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if (dtrm <= std::numeric_limits<double>::epsilon()) {
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }
              
            // 协方差矩阵的逆和行列式
            calcInverseCovAndDeterm(ci);
        }
    }
}
  
// 计算第ci个高斯分布的协方差矩阵的逆和行列式
void GMM::calcInverseCovAndDeterm(int ci) {
    if (coefs[ci] > 0) {
        // 获得第ci个高斯分布协方差的起始指针
        double *c = cov + 9 * ci;
        double dtrm = covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
  
        // 保证行列式大于0
        CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
        // 三阶方阵的求逆
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}
