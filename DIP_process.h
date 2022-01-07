#pragma once
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric> // 用以精确数值计算
#include <algorithm>

using namespace std;
using namespace cv;

// 图像处理类ImgProcess，所有个人题目与小组算法实现函数均声明在此头文件中
// 个人题目函数实现代码在DIP_process.cpp内，小组算法实现函数代码实现在DIP_process_group.cpp内
class ImgProcess
{
private:
	string savePath; // 原图像保存路径
	Mat M; // opencv读取的图像矩阵
	int Cols; // 图像列数，即高
	int Rows; // 图像行数，即宽
	int bands; // 波段数，默认3
	int pixels; // 表示图像的像素总量，p=c*r*b
	unsigned char* ptr; // 指向原图data的指针
	void SaveImage(Mat &imgM, string imageName); // 展示成果图片并保存，静态函数

	/* 这里是一些处理用算子的初始化 */
	vector<double> highPTemplate1{ 0,-1,0,-1,5,-1,0,-1,0 }; // 拉普拉斯算子
	vector<double> highPTemplate2{ 1,-2,1,-2,5,-2,1,-2,1 }; // 拉普拉斯八向高通滤波算子
	vector<double> highPTemplate3{ 0,0,-1,0,0,0,-1,-2,-1,0,-1,-2,16,-2,-1,0,-1,-2,-1,0,0,0,-1,0,0 }; // 5x5拉普拉斯算子
	vector<double> lowPTemplate{ 0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,
		0.11111111111,0.11111111111,0.11111111111,0.11111111111 }; //低通滤波算子

public:
	ImgProcess(string imgName); // 类构造函数
	ImgProcess(string imgName,int type); // 指定读取类型的类构造函数

	/*下面是完成的个人必做题和选做题和小组算法函数*/

	// 必做1：图像点运算：灰度线性变换
	void grayLinearTransfer(double k, double b);

	// 必做2：图像局部处理：高通滤波、低通滤波、中值滤波
	void PassFilters(int type); // 高通和低通滤波
	void midPassFilter(); // 中值滤波

	// 选做2：图像二值化：状态法及判断分析法
	void imgStateBin(); // 状态法
	void imgAnalysisBin(); // 判断分析法

	// 选做4：直方图匹配
	Mat equalize_hist(Mat& M); // 直方图均衡化
	void DrawgrayHist(Mat& M); // 绘制灰度直方图
	void histogramMatching(ImgProcess &matcher); // 即直方图规定化

	// 选做5：色彩平衡
	void whiteBalance(); // 色彩平衡：即白平衡
	void colorBalance(int deltaR, int deltaG, int deltaB); // 自定义色彩平衡

	///////////////////////////////////////////////////////

	// 小组指定算法：7、图像的频域处理
	// pCFData为频域数据，pCTData为时域数据，Width、Height为图像数据宽度和高度，nRadius是半功率点

	void ShowSpectrumMap(string name, Complex<double>* pCData, int nTransWidth, int nTransHeight); // 显示频谱
	void DFT(Complex<double>* pCTData, int Width, int Height, Complex<double>* pCFData); // 傅里叶变换
	void IDFT(Complex<double>* pCFData, int Width, int Height, Complex<double>* pCTData); // 傅里叶反变换
	void FFT(Complex<double>* pCTData, Complex<double>* pCFData, int nLevel);  // 快速傅里叶变换

	// 预处理，包括滤波开始前的大部分处理，即一次DFT与中心化等操作
	void PreHandle(ImgProcess& img, Complex<double>* &pCTData, Complex<double>* &pCFData, int nTransWidth, int nTransHeight);
	// 终处理，包括中心化还原，傅里叶反变换和反传数据、输出保存图像
	void LastHandle(ImgProcess& img, Complex<double>*& pCTData, Complex<double>*& pCFData, int nTransWidth, int nTransHeight, string Mname);

	void ILPF(ImgProcess& img, int nRadius); // 理想低通滤波
	void IHPF(ImgProcess& img, int nRadius); // 理想高通滤波

	void BLPF(ImgProcess& img, int nRadius); // 巴特沃斯低通滤波
	void BHPF(ImgProcess& img, int nRadius); // 巴特沃斯高通滤波

	void ELPF(ImgProcess& img, int nRadius); // 指数低通滤波
	void EHPF(ImgProcess& img, int nRadius); // 指数高通滤波

	void TLPF(ImgProcess& img, int nRadius1, int nRadius2); // 梯形低通滤波
	void THPF(ImgProcess& img, int nRadius1, int nRadius2); // 梯形高通滤波

	void HomomorphicFilter(ImgProcess& img, double gammaH, double gammaL); // 同态滤波

	///////////////////////////////////////////////////////

	/*
	* TODO
	// 选做1：图像的几何处理：平移、缩放、旋转
	void imgMove(int dx, int dy); // 图像平移
	void imgZoom(double xRate, double yRate); // 图像缩放
	void imgRotate(double angle); // 图像旋转

	// 选做3：纹理图像的自相关函数分析法
	void imgAutoCorr(); // 图像自相关函数：仅分析灰度图像，速度较慢
	*/

};