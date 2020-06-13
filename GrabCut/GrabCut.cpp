#include "GrabCut.h"

GrabCut2D::~GrabCut2D(void) {}

void GrabCut2D::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect, InputOutputArray _bgdModel, InputOutputArray _fgdModel, int iterCount, int mode) {
    
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();

    if (img.empty())
        CV_Error(Error::StsBadArg, "image is empty");
    if (img.type() != CV_8UC3)
        CV_Error(Error::StsBadArg, "image mush have CV_8UC3 type");

    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
    Mat compIdxs(img.size(), CV_32SC1);

    if (mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK) {
        initGMMs(img, mask, bgdGMM, fgdGMM);
    }

    if (iterCount <= 0)
        return;

    const double gamma = 50;
    const double lambda = 9 * gamma;
    const double beta = calcBeta(img);

    Mat leftW, upleftW, upW, uprightW;
    calcSmoothTerm(img, leftW, upleftW, upW, uprightW, beta, gamma);

    for (int i = 0; i < iterCount; i++) {
        detail::GCGraph<double> graph;
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
        estimateSegmentation(graph, mask);
    }
}

double GrabCut2D::calcBeta(const Mat& img) {
    double beta = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            // 计算邻域像素的欧氏距离
            Vec3d color = (Vec3d)img.at<Vec3b>(y, x);
            if (x > 0) {  // Left
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x-1);
                beta += diff.dot(diff);
            }
            if (y > 0 && x > 0) {  // Upleft
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1, x-1);
                beta += diff.dot(diff);
            }
            if (y > 0) {  // Up
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1, x);
                beta += diff.dot(diff);
            }
            if (y > 0 && x < img.cols - 1) {  // Upright
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1, x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.0f / (2 * beta / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows + 2));
  
    return beta;
}

void GrabCut2D::calcSmoothTerm(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma) {
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = (Vec3d)img.at<Vec3b>(y, x);
            if (x > 0) {  // Left
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x-1);
                leftW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
                CV_Assert(leftW.at<double>(y, x) >= 0);
            }
            else leftW.at<double>(y, x) = 0;
            if (x > 0 && y > 0) {  // Upleft
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1, x-1);
                upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
                CV_Assert(leftW.at<double>(y, x) >= 0);
            }
            else upleftW.at<double>(y, x) = 0;
            if (y > 0) {  // Up
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1, x);
                upW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
                CV_Assert(leftW.at<double>(y, x) >= 0);
            }
            else upW.at<double>(y, x) = 0;
            if (y > 0 && x < img.cols - 1) {  // Upright
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1, x+1);
                uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
                CV_Assert(leftW.at<double>(y, x) >= 0);
            }
            else uprightW.at<double>(y, x) = 0;
        }
    }
}

// 使用kmeans初始化GMM
void GrabCut2D::initGMMs(const Mat& _img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
    const int kMeansItCount = 10;  // 迭代次数
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for (p.y = 0; p.y < _img.rows; p.y++) {
        for (p.x = 0; p.x < _img.cols; p.x++) {
            if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                bgdSamples.push_back((Vec3f)_img.at<Vec3b>(p));
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back((Vec3f)_img.at<Vec3b>(p));
        }
    }
    CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
    
    // kmeans聚类
    Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
    kmeans(_bgdSamples, GMM::componentsCount, bgdLabels, TermCriteria(TermCriteria::COUNT, kMeansItCount, 0.0), 0, kMeansType);
    Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
    kmeans(_fgdSamples, GMM::componentsCount, fgdLabels, TermCriteria(TermCriteria::COUNT, kMeansItCount, 0.0), 0, kMeansType);

    // 训练高斯混合模型
    bgdGMM.initLearning();
    for (int i = 0; i < (int)bgdSamples.size(); i++)
        bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for (int i = 0; i < (int)fgdSamples.size(); i++)
        fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
    fgdGMM.endLearning();
}

// 计算每个像素所属的高斯分布
void GrabCut2D::assignGMMsComponents(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs) {
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            Vec3d color = img.at<Vec3b>(p);
            // 先判断该像素属于背景像素还是前景像素，再判断属于哪个高斯分布
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ? bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

// 根据像素集更新GMM参数
void GrabCut2D::learnGMMs(const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM) {
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for (int ci = 0; ci < GMM::componentsCount; ci++) {
        for (p.y = 0; p.y < img.rows; p.y++) {
            for (p.x = 0; p.x < img.cols; p.x++) {
                if (compIdxs.at<int>(p) == ci) {
                    if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                        bgdGMM.addSample(ci, img.at<Vec3b>(p));
                    else
                        fgdGMM.addSample(ci, img.at<Vec3b>(p));
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

// 构建图
void GrabCut2D::constructGCGraph(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda, const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW, detail::GCGraph<double>& graph) {
    int vtxCount = img.cols * img.rows;  // 顶点数
    int edgeCount = 2 * (4 * vtxCount - 3 * (img.cols + img.rows) + 2);  // 边数
    graph.create(vtxCount, edgeCount);
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // 顶点
            int vtxIdx = graph.addVtx();  // 返回顶点在图中的索引
            Vec3d color = img.at<Vec3b>(p);

            double fromSource, toTarget;
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                fromSource = -log(bgdGMM(color));
                toTarget = -log(fgdGMM(color));
            }
            else if (mask.at<uchar>(p) == GC_BGD) {
                // 对于确定为背景的像素点，它与Source点的连接为0，与Target点的连接为lambda
                fromSource = 0;
                toTarget = lambda;
            }
            else {  // GC_FGD
                fromSource = lambda;
                toTarget = 0;
            }
            
            // 设置该顶点分别与Source点和Sink点的连接权值
            graph.addTermWeights(vtxIdx, fromSource, toTarget);

            // 计算两个邻域顶点之间连接的权值
            if (p.x > 0) {
                double w = leftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx-1, w, w);
            }
            if (p.x > 0 && p.y > 0) {
                double w = upleftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx-img.cols-1, w, w);
            }
            if (p.y > 0) {
                double w = upW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx-img.cols, w, w);
            }
            if (p.x < img.cols-1 && p.y > 0) {
                double w = uprightW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx-img.cols+1, w, w);
            }
        }
    }
}

// 通过最大流算法确定图的最小割，完成图像的分割
void GrabCut2D::estimateSegmentation(detail::GCGraph<double>& graph, Mat& mask) {
    graph.maxFlow();
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {
            // 更新mask
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                if (graph.inSourceSegment(p.y*mask.cols+p.x /*vertex index*/ ))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}
