// DIP_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <stdio.h>
#include "DIP_process.h"

using namespace std;
using namespace cv;  // 省去函数前面加cv::的必要性

/*
* 头文件：DIP_process.h
* 函数实现文件：DIP_process.cpp, DIP_process_group.cpp
* main函数：DIP_test.cpp

个人必做题
1）图像点运算：灰度线性变换 √
2）图像局部处理：高通滤波、低通滤波、中值滤波 √

个人选做题
1）图像的几何处理：平移、缩放、旋转
2）图像二值化：状态法及判断分析法 √
3）纹理图像的自相关函数分析法
4）直方图匹配 √
5）色彩平衡 √

小组指定算法
7）图像的频域处理 √
*/

int main()
{
    /* 相对路径打开4张图像，图片存放在与include文件夹同级创建的images文件夹中 */
    ImgProcess Img1("../images/20180620-tianjin.bmp"); // 机场图
    ImgProcess Img2("../images/ik_beijing_c.bmp"); // 彩色模糊图
    ImgProcess Img3("../images/ik_beijing_p.bmp"); // 灰度图
    ImgProcess Img4("../images/lena.jpeg", IMREAD_GRAYSCALE); // Lena图
    
    /* 选择图像操作 */
    cout << "\n----------请选择您想进行的操作----------" << endl;
    cout << "1.（必做1）图像点运算：灰度线性变换" << endl;
    cout << "2.（必做2）图像局部处理：高通、低通、中值滤波" << endl;
    cout << "3.（选做2）图像二值化：状态法及判断分析法" << endl;
    cout << "4.（选做4）直方图匹配" << endl;
    cout << "5.（选做5）色彩平衡" << endl;
    cout << "6.（小组7）图像的频域处理" << endl;
    cout << "0. 退出程序" << endl;

    int k = 0, b = 0, choose = 0; // 初始化一些使用值
    int i = -1;
    while (i != 0) {
        cout << endl << ">>>>>>请输入您想进行的操作序号：";
        cin >> i;
        cout << endl;
        switch (i)
        {
        case 1:
            cout << "开始进行图像点运算，即灰度线性运算" << endl;
            cout << "请依次输入灰度线性变换的k值与b值：";
            cin >> k >> b;
            Img3.grayLinearTransfer(k, b);
            cout << "\n图像点运算，即灰度线性运算程序结束\n";
            break;

        case 2:
            cout << "开始进行图像局部处理：请选择您的处理方式：" << endl;
            cout << "1.高通滤波 2.低通滤波 3.中值滤波 4.退出\n" << "您选择的序号是：";
            cin >> choose;
            if (choose == 1) {
                Img2.PassFilters(1);
                Img2.PassFilters(2);
                Img2.PassFilters(3);
            }
            else if (choose == 2) {
                Img2.PassFilters(4);
            }
            else if (choose == 3) {
                Img2.midPassFilter();
            }
            else {
                cout << "您选择退出，图像局部处理程序结束" << endl;
            }
            break;

        case 3:
            cout << "开始进行图像二值化处理：请选择您的处理方式：" << endl;
            cout << "1.状态法 2.判断分析法 3.退出\n" << "您选择的序号是：";
            cin >> choose;
            if (choose == 1) {
                Img1.imgStateBin();
            }
            else if (choose == 2) {
                Img1.imgAnalysisBin();
            }
            else {
                cout << "您选择退出，图像二值化处理程序结束" << endl;
            }
            break;

        case 4:
            cout << "开始进行图像直方图匹配处理";
            Img1.histogramMatching(Img2);
            cout << "\n图像直方图匹配结束\n";
            break;

        case 5:
            cout << "开始进行图像色彩平衡处理，请选择色彩平衡的方式" << endl;
            cout << "1.白平衡 2.自定义 3.退出\n" << "您选择的方式序号是：";
            cin >> choose;
            if (choose == 1) {
                Img1.whiteBalance();
            }
            else if (choose == 2) {
                cout << endl << "\n请依次按顺序输入RGB改变的值" << endl;
                cin >> k >> b >> choose;
                Img1.colorBalance(k, b, choose);
            }
            else {
                cout << "您选择退出，图像色彩平衡处理程序结束" << endl;
            }
            break;

        case 6:
            cout << "开始进行小组算法：图像的频域处理：请选择您的处理方式：" << endl;
            cout << "1.频率域图像平滑 2.频率域图像锐化 3.同态滤波 4.退出\n" << "您选择的序号是：";
            cin >> choose;
            if (choose == 1) {
                cout << "1.理想低通滤波器 2.巴特沃斯低通滤波器 3.指数低通滤波器 4.梯形低通滤波器 5.退出\n";
                cout << "请选择您的频率域平滑滤波器：";
                cin >> k;
                if (k == 1) {
                    Img4.ILPF(Img4, 80);
                }
                else if (k == 2) {
                    Img4.BLPF(Img4, 80);
                }
                else if (k == 3) {
                    Img4.ELPF(Img4, 80);
                }
                else if (k == 4) {
                    Img4.TLPF(Img4, 10, 20);
                }
                else {
                    cout << "您选择退出，频率域图像平滑处理程序结束" << endl;
                }
            }
            else if (choose == 2) {
                cout << "1.理想高通滤波器 2.巴特沃斯高通滤波器 3.指数高通滤波器 4.梯形高通滤波器 5.退出\n";
                cout << "请选择您的频率域锐化滤波器：";
                cin >> k;
                if (k == 1) {
                    Img4.IHPF(Img4, 3);
                }
                else if (k == 2) {
                    Img4.BHPF(Img4, 5);
                }
                else if (k == 3) {
                    Img4.EHPF(Img4, 1);
                }
                else if (k == 4) {
                    Img4.THPF(Img4, 10, 40);
                }
                else {
                    cout << "您选择退出，频率域图像锐化处理程序结束" << endl;
                }
            }
            else if (choose == 3) {
                cout << "您选择高斯同态滤波滤波器";
                Img4.HomomorphicFilter(Img4, 1.5, 0.5);
            }
            else {
                cout << "您选择退出，图像的频域处理程序结束" << endl;
            }
            break;

        default:
            cout << "您没有选择进行任何操作，退出程序！";
            return 0;
        }
    }
    cout << "主程序已结束！";
}