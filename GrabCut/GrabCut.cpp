#include "GrabCut.h"

GrabCut2D::~GrabCut2D(void) {}

void GrabCut2D::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect, InputOutputArray _bgdModel, InputOutputArray _fgdModel, int iterCount, int mode) {
    
    std::cout << "Execute GrabCut Function: Please finish the code here!" << std::endl;

//一.参数解释：
	//输入：
    //InputArray _img,     :输入的color图像(类型-cv:Mat)
    //Rect rect            :在图像上画的矩形框（类型-cv:Rect)
  	//int iterCount :           :每次分割的迭代次数（类型-int)


	//中间变量
	//InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）


	//输出:
	//InputOutputArray _mask  : 输出的分割结果 (类型： Mat)

//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	//4.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
}
