#pragma once
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric> // ���Ծ�ȷ��ֵ����
#include <algorithm>

using namespace std;
using namespace cv;

// ͼ������ImgProcess�����и�����Ŀ��С���㷨ʵ�ֺ����������ڴ�ͷ�ļ���
// ������Ŀ����ʵ�ִ�����DIP_process.cpp�ڣ�С���㷨ʵ�ֺ�������ʵ����DIP_process_group.cpp��
class ImgProcess
{
private:
	string savePath; // ԭͼ�񱣴�·��
	Mat M; // opencv��ȡ��ͼ�����
	int Cols; // ͼ������������
	int Rows; // ͼ������������
	int bands; // ��������Ĭ��3
	int pixels; // ��ʾͼ�������������p=c*r*b
	unsigned char* ptr; // ָ��ԭͼdata��ָ��
	void SaveImage(Mat &imgM, string imageName); // չʾ�ɹ�ͼƬ�����棬��̬����

	/* ������һЩ���������ӵĳ�ʼ�� */
	vector<double> highPTemplate1{ 0,-1,0,-1,5,-1,0,-1,0 }; // ������˹����
	vector<double> highPTemplate2{ 1,-2,1,-2,5,-2,1,-2,1 }; // ������˹�����ͨ�˲�����
	vector<double> highPTemplate3{ 0,0,-1,0,0,0,-1,-2,-1,0,-1,-2,16,-2,-1,0,-1,-2,-1,0,0,0,-1,0,0 }; // 5x5������˹����
	vector<double> lowPTemplate{ 0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,
		0.11111111111,0.11111111111,0.11111111111,0.11111111111 }; //��ͨ�˲�����

public:
	ImgProcess(string imgName); // �๹�캯��
	ImgProcess(string imgName,int type); // ָ����ȡ���͵��๹�캯��

	/*��������ɵĸ��˱������ѡ�����С���㷨����*/

	// ����1��ͼ������㣺�Ҷ����Ա任
	void grayLinearTransfer(double k, double b);

	// ����2��ͼ��ֲ�������ͨ�˲�����ͨ�˲�����ֵ�˲�
	void PassFilters(int type); // ��ͨ�͵�ͨ�˲�
	void midPassFilter(); // ��ֵ�˲�

	// ѡ��2��ͼ���ֵ����״̬�����жϷ�����
	void imgStateBin(); // ״̬��
	void imgAnalysisBin(); // �жϷ�����

	// ѡ��4��ֱ��ͼƥ��
	Mat equalize_hist(Mat& M); // ֱ��ͼ���⻯
	void DrawgrayHist(Mat& M); // ���ƻҶ�ֱ��ͼ
	void histogramMatching(ImgProcess &matcher); // ��ֱ��ͼ�涨��

	// ѡ��5��ɫ��ƽ��
	void whiteBalance(); // ɫ��ƽ�⣺����ƽ��
	void colorBalance(int deltaR, int deltaG, int deltaB); // �Զ���ɫ��ƽ��

	///////////////////////////////////////////////////////

	// С��ָ���㷨��7��ͼ���Ƶ����
	// pCFDataΪƵ�����ݣ�pCTDataΪʱ�����ݣ�Width��HeightΪͼ�����ݿ�Ⱥ͸߶ȣ�nRadius�ǰ빦�ʵ�

	void ShowSpectrumMap(string name, Complex<double>* pCData, int nTransWidth, int nTransHeight); // ��ʾƵ��
	void DFT(Complex<double>* pCTData, int Width, int Height, Complex<double>* pCFData); // ����Ҷ�任
	void IDFT(Complex<double>* pCFData, int Width, int Height, Complex<double>* pCTData); // ����Ҷ���任
	void FFT(Complex<double>* pCTData, Complex<double>* pCFData, int nLevel);  // ���ٸ���Ҷ�任

	// Ԥ���������˲���ʼǰ�Ĵ󲿷ִ�����һ��DFT�����Ļ��Ȳ���
	void PreHandle(ImgProcess& img, Complex<double>* &pCTData, Complex<double>* &pCFData, int nTransWidth, int nTransHeight);
	// �մ����������Ļ���ԭ������Ҷ���任�ͷ������ݡ��������ͼ��
	void LastHandle(ImgProcess& img, Complex<double>*& pCTData, Complex<double>*& pCFData, int nTransWidth, int nTransHeight, string Mname);

	void ILPF(ImgProcess& img, int nRadius); // �����ͨ�˲�
	void IHPF(ImgProcess& img, int nRadius); // �����ͨ�˲�

	void BLPF(ImgProcess& img, int nRadius); // ������˹��ͨ�˲�
	void BHPF(ImgProcess& img, int nRadius); // ������˹��ͨ�˲�

	void ELPF(ImgProcess& img, int nRadius); // ָ����ͨ�˲�
	void EHPF(ImgProcess& img, int nRadius); // ָ����ͨ�˲�

	void TLPF(ImgProcess& img, int nRadius1, int nRadius2); // ���ε�ͨ�˲�
	void THPF(ImgProcess& img, int nRadius1, int nRadius2); // ���θ�ͨ�˲�

	void HomomorphicFilter(ImgProcess& img, double gammaH, double gammaL); // ̬ͬ�˲�

	///////////////////////////////////////////////////////

	/*
	* TODO
	// ѡ��1��ͼ��ļ��δ���ƽ�ơ����š���ת
	void imgMove(int dx, int dy); // ͼ��ƽ��
	void imgZoom(double xRate, double yRate); // ͼ������
	void imgRotate(double angle); // ͼ����ת

	// ѡ��3������ͼ�������غ���������
	void imgAutoCorr(); // ͼ������غ������������Ҷ�ͼ���ٶȽ���
	*/

};