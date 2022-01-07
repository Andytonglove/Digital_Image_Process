#include "DIP_process.h"

void ImgProcess::SaveImage(Mat &imgM, string imageName)
{
    /* 本函数完成图像的保存及显示 */
    if (imgM.empty()) {
        cout << "图像存在问题，无法正确保存！"; // 检验错误
        waitKey(6000);
    }
    namedWindow("image", 0); // 1为按图片大小显示，0为跟据窗口大小调整
    imshow("image", imgM); // 显示图片
    waitKey(); // 这里的waitkey是必要的，不然会跳过或卡死
    imwrite(imageName, imgM); // 保存图片
}

ImgProcess::ImgProcess(string imageName)
{
    /* 构造函数：打开图像并进行初步处理 */
    M = imread(imageName); // 读入图片，由程序自行判断是color还是grayscale

    // 判断文件是否正常打开！
    if (M.empty())
    {
        cout << "无法打开图像！请检查文件路径！\n";
        return;
    }
    cout << "图像打开成功！可以进行下一步处理操作！\n";

    //初始化类中所有变量
    this->Cols = M.cols;
    this->Rows = M.rows;
    this->bands = M.channels();
    this->pixels = Cols * Rows * bands;
    this->ptr = M.data;
}

ImgProcess::ImgProcess(string imgName,int type)
{
    /* 构造函数：指定读取文件类型的构造函数 */
    M = imread(imgName, type); // 读入图片，由程序自行判断是color还是grayscale

    // 判断文件是否正常打开！
    if (M.empty())
    {
        cout << "无法打开图像！请检查文件路径！\n";
        return;
    }
    cout << "图像打开成功！可以进行下一步处理操作！\n";

    //初始化类中所有变量
    this->Cols = M.cols;
    this->Rows = M.rows;
    this->bands = M.channels();
    this->pixels = Cols * Rows * bands;
    this->ptr = M.data;
}

/* 下面的函数对图像进行点运算：灰度线性变换 */
void ImgProcess::grayLinearTransfer(double k, double b)
{
    /* 这里本函数对图像进行灰度线性变换 */
    Mat grayImg = Mat::zeros(M.size(), M.type()); // 初始化图像大小

    //灰度变换
    for (int r = 0; r < Rows; r++) {
        for (int c = 0; c < Cols; c++) {
            if (M.channels() == 3) {
                // Vec3b，表示对彩色图像进行处理，这里颜色是按照b，g，r反序排序的
                // saturate_cast 处理！指定数据类型的溢出，保证了图像在进行灰度线性变换后不会产生色彩溢出
                // 解决数据值超限问题0-256
                grayImg.at<Vec3b>(r, c)[0] = saturate_cast<uchar>((k * M.at<Vec3b>(r, c)[0] + b)); // b
                grayImg.at<Vec3b>(r, c)[1] = saturate_cast<uchar>((k * M.at<Vec3b>(r, c)[1] + b)); // g
                grayImg.at<Vec3b>(r, c)[2] = saturate_cast<uchar>((k * M.at<Vec3b>(r, c)[2] + b)); // r
            }else if (M.channels() == 1) {
                // uchar对灰度图像进行处理
                int v = M.at<uchar>(r, c);
                grayImg.at<uchar>(r, c) = saturate_cast<uchar>(k * v + b);
            }
            else {
                cout << "不支持本种类型图像的灰度变换！" << endl;
            }
        }
    }

    /*调用函数保存图片，特别注意opencv这里只能特别设定保存为位图格式，否则会报错，很奇怪*/
    SaveImage(grayImg, "灰度线性变换图像.bmp");
}

/* 下面的2个函数对图像进行局部处理（高通/低通、中值滤波）处理 */
void ImgProcess::PassFilters(int type)
{
    /* 本函数对图像进行各类滤波处理 */
    Mat passImg = Mat::zeros(M.size(), M.type());
    unsigned char* pDst = passImg.data;
    vector<double> _template; // 初始化
    string args = "";

    /* 滤波选择 */
    switch (type)
    {
    case 1:
        _template = highPTemplate1;
        args = "高通滤波1.bmp";
        break;
    case 2:
        _template = highPTemplate2;
        args = "高通滤波2.bmp";
        break;
    case 3:
        _template = highPTemplate3;
        args = "拉普拉斯高通滤波3.bmp";
        break;
    case 4:
        _template = lowPTemplate;
        args = "低通滤波.bmp";
        break;
    default:
        /*中值滤波需要用另外的方式更好处理*/
        cout << "没有此种滤波，程序退出！";
        break;
    }

    int tempSize = sqrt(_template.size()); // 读取模板总大小
    int directSize = (tempSize - 1) / 2; // 用于计算的模板的方向延展范围
    vector<int> myPixels; // 为模板涉及到的内存开辟一个vector数组，使用vector容器避免内存分配问题
    int pixelNum = 0; // 当前处理的是第n个像素
    int totalpixel = 0; // 像素总灰度级

    // 首先，对于n通道，其计算范围为行、列中除开小于等于延展范围的边缘部分
    for (int i = 0; i < bands; i++) {
        for (int j = directSize; j < Rows - directSize; j++) {
            for (int k = directSize; k < Cols - directSize; k++) {
                // 邻域处理
                for (int m = j - directSize; m <= j + directSize; m++) {
                    for (int n = k - directSize; n <= k + directSize; n++) {
                        // vector数组末尾不断进入元素，即模板的第size个元素乘以对应位置的像素值
                        myPixels.push_back(_template[myPixels.size()] * ptr[(m * Cols + n) * bands + i]);
                    }
                }
                totalpixel = accumulate(myPixels.begin(), myPixels.end(), 0);// 累加各个像素
                myPixels.clear(); // 清空临时的像素数组
                // 限制灰度级使之不溢出，使其最大范围在原范围中
                pDst[(j * Cols + k) * bands + i] = saturate_cast<uchar>(totalpixel);
            }
        }
    }
    SaveImage(passImg, args);
}

void ImgProcess::midPassFilter()
{
    /* 本函数对图像进行中值滤波处理，以默认3*3为例 */
    Mat passImg = Mat::zeros(M.size(), M.type());
    unsigned char* pNtr = passImg.data;
    int const num = 3 * 3;
    uchar pixel[num] = { 0 }; // 保存邻域的像素值

    //相对于中心点，3*3不去心领域中的点需要偏移的位置，相当于一个位置偏移矩阵！
    int delta[3 * 3][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1} };

    for (int k = 0; k < bands; k++) {
        for (int i = 1; i < Rows - 1; i++) {
            for (int j = 1; j < Cols - 1; j++) {
                // 提取3*3邻域值
                for (int c = 0; c < num; ++c)
                {
                    pixel[c] = ptr[((i + delta[c][0]) * Cols + (j + delta[c][1])) * bands + k];
                }
                sort(pixel, pixel + 9); // 利用sort函数排序找到中间值
                pNtr[(i * Cols + j) * bands + k] = pixel[5];
            }
        }
    }
    SaveImage(passImg, "中值滤波.bmp");
}

/* 下面的2个函数对图像进行二值化处理 */
void ImgProcess::imgStateBin()
{
    /* 状态法（峰谷法） */
    int T = 0; // 阈值
    int nNewThre = 0;
    //初始化定义类1、类2像素总数；类1、类2灰度均值
    int cnt1 = 0, cnt2 = 0, mval1 = 0, mval2 = 0;
    int iter = 0; // 迭代次数
    // 先将彩色图像化为灰度图像
    imwrite("pic_prebin.bmp", M);
    Mat binImg = imread("pic_prebin.bmp", IMREAD_GRAYSCALE);
    DrawgrayHist(M);

    int nEISize = binImg.elemSize(); // 获取每个像素的字节数
    int G = pow(2, double(8 * nEISize)); // 灰度级数
    nNewThre = int(G / 2); // 给阈值赋迭代初值
    // 分配灰度级数个量的内存，储存并计算灰度直方图
    auto* hist = new int[G]; // 灰度统计数组
    for (int i = 0; i < G; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            int g = binImg.at<uchar>(i, j);
            hist[g]++;
        }
    }
    // 迭代求最佳阈值
    for (iter = 0;iter < 100;iter++) {
        /* 初始化 */
        T = nNewThre;
        for (int m = T; m < G; m++) {
            cnt2 += hist[m];
        }
        cnt1 = Cols * Rows - cnt2;
        // 组1从0开始计数到小于i，组2反向计数从G递减到等于i
        for (int n = T; n < G; n++) {
            mval2 += (double(hist[n]) / cnt2) * n;
        }
        for (int k = 0; k < T; k++) {
            mval1 += (double(hist[k]) / cnt1) * k;
        }
        T = int(mval1 + mval2) / 2; //得新阈值
    }
    for (int i = 0;i < Rows;i++) {
        for (int j = 0;j < Cols;j++) {
            binImg.at<uchar>(i, j) = (binImg.at<uchar>(i, j) > T) ? (G - 1) : 0; // 特别注意这里最大像素值是G-1！！！
        }
    }
    SaveImage(binImg, "状态法二值化.bmp");

    delete[]hist;
}

void ImgProcess::imgAnalysisBin()
{
    /* 判断分析法：基本原理：使被阈值区分的两组灰度级之间，组内组间方差比最大 */
    /* 一些值的定义与初始化 */
    double ratio = 0;
    int Thre = 0; // 阈值
    double cnt1 = 0, cnt2 = 0; // 两组像素的总数量
    double mval1 = 0, mval2 = 0, mval = 0; // 两组像素的灰度平均值以及整幅图像的灰度平均值
    double delta1 = 0, delta2 = 0; // 第一组、第二组像素的方差
    double deltaW = 0, deltaB = 0; // 组内方差、组间方差

    // 先将彩色图像化为灰度图像
    imwrite("pic_prebin.bmp", M);
    Mat binImg = imread("pic_prebin.bmp", IMREAD_GRAYSCALE);
    DrawgrayHist(M);

    int nEISize = binImg.elemSize(); // 获取每个像素的字节数
    int G = pow(2, double(8 * nEISize)); // 灰度级数
    // 分配灰度级数个量的内存，储存并计算灰度直方图
    auto* hist = new int[G]; // 灰度统计数组
    for (int i = 0; i < G; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            int H = binImg.at<uchar>(i, j);
            hist[H]++;
        }
    }

    // 通过使组内方差与组间方差之比最大来确定阈值T
    for (int i = 0; i < G; i++) {
        for (int m = i; m < G; m++) {
            cnt2 += hist[m];
        }
        cnt1 = Cols * Rows - cnt2;

        // 组1从0开始计数到小于i，组2反向计数从G递减到等于i
        for (int n = i; n < G; n++) {
            mval2 += (double(hist[n]) / cnt2) * n;
        }
        for (int k = 0; k < i; k++) {
            mval1 += (double(hist[k]) / cnt1) * k;
        }
        
        // 整幅图像的灰度平均值计算
        mval = (mval1 * cnt1 + mval2 * cnt2) / (cnt1 + cnt2);

        // 两组的方差以及组内方差和组间方差计算，同上理
        for (int p = i; p < G; p++) {
            delta2 += (double(hist[p]) / cnt2) * pow((p - mval2), 2);
        }
        for (int q = 0;q < i;q++) {
            delta1 += (double(hist[q]) / cnt1) * pow((q - mval1), 2);
        }
        deltaW = cnt1 * delta1 + cnt2 * delta2;
        deltaB = cnt1 * cnt2 * pow((mval1 - mval2), 2);
        if ((deltaB / deltaW) > ratio) {
            ratio = deltaB / deltaW;
            Thre = i; // 阈值T计算
        }
        // 重新赋值还原为0
        cnt1 = 0;
        cnt2 = 0;
        mval1 = 0;
        mval2 = 0;
        delta1 = 0;
        delta2 = 0;
        deltaW = 0;
        deltaB = 0;
    }

    // 根据阈值进行二值化处理
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            binImg.at<uchar>(i, j) = (binImg.at<uchar>(i, j) < Thre) ? 0 : (G - 1); // 通过阈值来赋值
        }
    }
    SaveImage(binImg, "判断分析法二值化.bmp");

    delete[]hist;
}

/* 下面的2个函数为直方图匹配 */
Mat ImgProcess::equalize_hist(Mat& M)
{
    /* 本函数用以直方图均衡化，返回均衡化后的矩阵，便于下一步计算 */
    Mat OutputM = M.clone(); // 返回的输出图像矩阵
    int gray[256] = { 0 }; // 记录每个灰度级别下的像素个数
    double gray_prob[256] = { 0 }; // 记录灰度分布密度
    double gray_distribution[256] = { 0 }; // 记录累计密度
    int gray_equal[256] = { 0 }; // 均衡化后的灰度值
    int gray_sum = M.cols * M.rows; // 像素总数

    // 统计每个灰度下的像素个数
    for (int i = 0; i < M.rows; i++)
    {
        uchar* ptr = M.ptr<uchar>(i);
        for (int j = 0; j < M.cols; j++)
        {
            int vaule = ptr[j];
            gray[vaule]++;
        }
    }

    // 统计灰度频率和累计密度
    for (int i = 0; i < 256; i++)
    {
        gray_prob[i] = ((double)gray[i] / gray_sum);
    }
    gray_distribution[0] = gray_prob[0];
    for (int i = 1; i < 256; i++)
    {
        gray_distribution[i] = gray_distribution[i - 1] + gray_prob[i];
    }

    // 重新计算均衡化后的灰度值，(N-1)*T+0.5，四舍五入
    for (int i = 0; i < 256; i++)
    {
        gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
    }

    // 直方图均衡化,更新原图每个点的像素值
    for (int i = 0; i < OutputM.rows; i++)
    {
        uchar* ptr = OutputM.ptr<uchar>(i);
        for (int j = 0; j < OutputM.cols; j++)
        {
            ptr[j] = gray_equal[ptr[j]];
        }
    }
    return OutputM;
}

void ImgProcess::DrawgrayHist(Mat& M)
{
    /* 本函数用以绘制灰度直方图 */
    // 额外绘制直方图，本函数为了方便中使用了部分cv函数
    int channels = 0; // 通道数
    MatND dstHist; // 直方图输出的结果存储空间，用MatND类型来存储结果  
    int histSize[] = { 256 }; // 直方图的每一个维度的柱条数
    float midRanges[] = { 0, 256 }; // 取值范围
    const float* ranges[] = { midRanges };
    calcHist(&M, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);
    // ps 这里为了方便，调用cv的函数来快速实现灰度直方图计算，不调用函数实现的核心代码已经完成在上面了

    Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3); // 创建一个8位的3通道图像
    // 同上，用cv2函数来得到直方图统计像素最大个数，进行范围限制
    double g_dHistMaxValue;
    minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);
    int size = 256; // 直方图尺寸
    //遍历直方图得到的数据绘图
    for (int i = 0; i < 256; i++)
    {
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);
        rectangle(drawImage, Point(i, size - 1),
            Point(i, size - value), Scalar(255, 255, 255)); // 绘制直方图
    }
    imshow("灰度直方图图像", drawImage);
    waitKey(); // 必须加上waitkey！不然会崩溃，不知原因
}

void ImgProcess::histogramMatching(ImgProcess &matcher)
{
    /* 本函数用以直方图规定化 */
    Mat image_matched = Mat::zeros(M.rows, M.cols, M.type());
    Mat lu_table(1, 256, CV_8U); // opencv提供了LUT的映射方法，这里自己实现一个简易的
    int* _gray = new int[256]; // 将要被规范化图象的原直方图
    int* gray = new int[256]; // 用以规定直方图图像的直方图
    for (int i = 0;i < 256;i++) {
        _gray[i] = 0;
        gray[i] = 0;
    }

    // 如果是彩色图像，那么先转化为灰度图像
    imwrite("pre_pic_this.bmp", this->M); // 这个是被规范化的！要用_gray[]！！！！
    Mat ImgM = imread("pre_pic_this.bmp", IMREAD_GRAYSCALE);
    imwrite("pre_pic_matcher.bmp", matcher.M);
    Mat ImgThis = imread("pre_pic_matcher.bmp", IMREAD_GRAYSCALE);
    image_matched = ImgM.clone();

    // 调用上一个函数完成直方图均衡化
    equalize_hist(ImgM);
    equalize_hist(ImgThis);

    // 经典循环分别计算均衡化后灰度数组
    for (int i = 0; i < ImgM.rows; i++)
    {
        uchar* ptr = ImgM.ptr<uchar>(i);
        for (int j = 0; j < ImgM.cols; j++)
        {
            int vaule = ptr[j];
            _gray[vaule]++;
        }
    }
    for (int i = 0; i < ImgThis.rows; i++)
    {
        uchar* ptr = ImgThis.ptr<uchar>(i);
        for (int j = 0; j < ImgThis.cols; j++)
        {
            int vaule = ptr[j];
            gray[vaule]++;
        }
    }
    // 算得概率差值，用于找到最接近的点，来进行匹配
    float(*differ)[256] = new float[256][256];// 直接分配栈容易溢出，需要考虑用堆
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++)
        {
            differ[i][j] = fabs(gray[i] - _gray[j]); // 谁前谁后无所谓
        }
    }
    for (int i = 0; i < 256; i++)
    {
        //找前后两个灰度中最接近的最小的值
        double min = differ[i][0];
        int index = 0;
        for (int j = 0; j < 256; j++) {
            if (min > differ[i][j]) {
                min = differ[i][j];
                index = j;
            }
        }
        lu_table.at<uchar>(i) = index;
    }
    // 图像中实现映射
    for (int i = 0;i < M.rows;i++) {
        uchar* ptr = ImgM.ptr<uchar>(i);
        uchar* pDst = image_matched.ptr<uchar>(i);
        for (int j = 0;j < M.cols;j++) {
            for (int g = 0;g < 256;g++) {
                if (ptr[j] == g) {
                    pDst[j] = lu_table.at<uchar>(g);
                }
            }
        }
    }
    SaveImage(image_matched, "直方图匹配.bmp");
    DrawgrayHist(image_matched); // 调用函数绘制灰度直方图

    delete[]gray;
    delete[]_gray;
    delete[]differ;
}

/* 下面的2个函数为色彩平衡 */
void ImgProcess::whiteBalance()
{
    /* 白平衡处理 */
    Mat dst;
    dst.create(M.size(), M.type());
    int HistRGB[767] = { 0 }; // 数组，三通道256*3=768个
    int MaxVal = 0; // RGB最值
    int Threshold = 0; // 阈值
    int sum = 0;
    float ratio = 0.1; // 比例
    int B_avg = 0, G_avg = 0, R_avg = 0, count = 0; // 计算三色和均值用

    // 统计得到R，G，B中的最大值MaxVal和三色和
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            MaxVal = max(MaxVal, (int)M.at<Vec3b>(i, j)[0]);
            MaxVal = max(MaxVal, (int)M.at<Vec3b>(i, j)[1]);
            MaxVal = max(MaxVal, (int)M.at<Vec3b>(i, j)[2]);
            sum = M.at<Vec3b>(i, j)[0] + M.at<Vec3b>(i, j)[1] + M.at<Vec3b>(i, j)[2];
            HistRGB[sum]++;
        }
    }
    // 计算三色和的数量超过像素总数比例的像素值，计算出阈值Threshold
    for (int i = 766; i >= 0; i--) {
        sum += HistRGB[i];
        if (sum > M.rows * M.cols * ratio) {
            Threshold = i;
            break;
        }
    }
    // 计算三色和大于阈值的所有点的均值
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            int sumP = M.at<Vec3b>(i, j)[0] + M.at<Vec3b>(i, j)[1] + M.at<Vec3b>(i, j)[2];
            if (sumP > Threshold) {
                B_avg += M.at<Vec3b>(i, j)[0];
                G_avg += M.at<Vec3b>(i, j)[1];
                R_avg += M.at<Vec3b>(i, j)[2];
                count++;
            }
        }
    }
    //得到均值
    int color[3] = { 0 }; // 0、1、2分别对应蓝绿红
    int avg[3] = { B_avg/count, G_avg/count, R_avg/count }; // 得到均值
    
    // 重新分配得到新矩阵dst
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            for (int k = 0;k < 3;k++) {
                color[k] = M.at<Vec3b>(i, j)[k] * MaxVal / avg[k]; // blue->green->red
                if (color[k] > 255 || color[k] < 0) {
                    color[k] = (color[k] > 255) ? 255 : 0;
                }
            }
            for (int k = 0;k < 3;k++) {
                dst.at<Vec3b>(i, j)[k] = color[k];
            }
        }
    }
    SaveImage(dst, "色彩平衡.bmp");
}

void ImgProcess::colorBalance(int deltaR, int deltaG, int deltaB)
{
    Mat colorBalanceImg; // 用于储存色彩平衡后的图像
    colorBalanceImg.create(M.size(), M.type());
    colorBalanceImg = cv::Scalar::all(0); // 初始化，全部赋0
    int delta[3] = { deltaB,deltaG,deltaR }; // opencv的RGB顺序是反的

    for (int i = 0; i < Rows; i++)
    {
        auto* pt = (uchar*)(M.data + M.step * i);
        auto* dst = (uchar*)colorBalanceImg.data + colorBalanceImg.step * i;
        for (int j = 0; j < Cols; j++)
        {
            for (int k = 0;k < bands; k++) {
                dst[j * bands + k] = saturate_cast<uchar>(pt[j * bands + k] + delta[k]);
            }
        }
    }
    SaveImage(colorBalanceImg, "自定义色彩平衡图像.bmp");
}

///////////////////////////////////////////////////////

/*
// 下面函数为图像平移、缩放和旋转操作与纹理图像的自相关函数分析法 TODO

void ImgProcess::imgMove(int dx, int dy)
{
    // 下面三个函数分别对图像进行平移、缩放和旋转（矩阵变换）操作
    //向右、下方分别平移（dx，dy）像素
    Mat movedImg;
    Vec3b* p; // 相当于uchar类型的,长度为3的vector向量

    // 改变总图像大小
    int rows = Rows + abs(dy); //输出图像的大小
    int cols = Cols + abs(dx);
    movedImg.create(rows, cols, M.type());
    for (int i = 0; i < rows; i++)
    {
        p = movedImg.ptr<Vec3b>(i); // 获取图像行指针
        for (int j = 0; j < cols; j++)
        {
            // 位置坐标映射新坐标->原坐标+位移量
            int x = j - dx;
            int y = i - dy;
            // 保证映射后的坐标在原图像范围内
            if (x >= 0 && y >= 0 && x < cols && y < rows)
                p[j] = M.ptr<Vec3b>(y)[x];
        }
    }
    SaveImage(movedImg, "图像平移变换.bmp");
}
// TODO
void ImgProcess::imgZoom(double xRate, double yRate)
{
    // 缩放图像后的矩阵大小
    int rows = Rows * xRate;
    int cols = Cols * yRate;
    Mat zoomedImg = Mat::zeros(rows, cols, M.type());
    uchar* pDst = zoomedImg.data;

    // 构造缩放变换矩阵
    Mat T = (Mat_<double>(3, 3) << xRate, 0, 0, 0, yRate, 0, 0, 0, 1);
    Mat invT = T.inv(); // 矩阵求逆

    // 注，指定像素显式构造了变换单位矩阵形如：{α, 0, 0; 0, β, 0; 0, 0, 1}
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1); // 第i行j列的像素坐标(j,i)
            Mat src_uv = dst_xy * invT;
            double u = src_uv.at<double>(0, 0); // 原图像的横坐标，对应图像的列数
            double v = src_uv.at<double>(0, 1);

            // 双线性插值法插值运算不存在的像素点
            if (u >= 0 && v >= 0 && u <= Cols - 1 && v <= Rows - 1) {
                // 与映射到四邻域像素点的坐标，分别向下向上取整
                int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u);
                // 存在坐标偏差，故dv、du分别为行、列小数部分
                double dv = v - top;
                double du = u - left;
                for (int k = 0; k < bands; k++) {
                    pDst[(i * cols + j) * bands + k] =
                        (1 - dv) * (1 - du) * ptr[(top * Cols + left) * bands + k]
                        + (1 - dv) * du * ptr[(top * Cols + right) * bands + k]
                        + dv * (1 - du) * ptr[(bottom * Cols + left) * bands + k]
                        + dv * du * ptr[(bottom * Cols + right) * bands + k];
                }
            }
        }
    }
    SaveImage(zoomedImg, "图像缩放变换.bmp");
}

void ImgProcess::imgRotate(double angle)
{
    // 旋转过程中，angle为顺时针旋转的角度
    angle = angle * CV_PI / 180;
    int rows = round(fabs(Rows * cos(angle)) + fabs(Cols * sin(angle)));
    int cols = round(fabs(Cols * cos(angle)) + fabs(Rows * sin(angle)));
    // 构造矩阵
    Mat rotImg = Mat::zeros(rows, cols, M.type());
    uchar* pDst = rotImg.data;

    //构造旋转变换矩阵，基本方法同旋转
    Mat T1 = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.5 * Cols, 0.5 * Rows, 1.0);
    Mat T2 = (Mat_<double>(3, 3) << cos(angle), -sin(angle), 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0, 1.0);
    // 笛卡尔坐标映射
    double t3[3][3] = { { 1.0, 0.0, 0.0 },{ 0.0, -1.0, 0.0 },{ 0.5 * rotImg.cols, 0.5 * rotImg.rows ,1.0 } };
    Mat T3 = Mat(3.0, 3.0, CV_64FC1, t3);
    Mat T = T1 * T2 * T3;
    Mat invT = T.inv();//求逆

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1); // 第i行j列的像素坐标(j,i)
            Mat src_uv = dst_xy * invT;
            double u = src_uv.at<double>(0, 0); // 原图像的横坐标，对应图像的列数
            double v = src_uv.at<double>(0, 1);

            //双线性插值法
            if (u >= 0 && v >= 0 && u <= Cols - 1 && v <= Rows - 1) {
                int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u);
                double dv = v - top;
                double du = u - left;
                for (int k = 0; k < bands; k++) {
                    pDst[(i * cols + j) * bands + k] =
                        (1 - dv) * (1 - du) * ptr[(top * Cols + left) * bands + k]
                        + (1 - dv) * du * ptr[(top * Cols + right) * bands + k]
                        + dv * (1 - du) * ptr[(bottom * Cols + left) * bands + k]
                        + dv * du * ptr[(bottom * Cols + right) * bands + k];
                }
            }
        }
    }
    SaveImage(rotImg, "图像旋转变换.bmp");
}

// 下面的函数为纹理图像的自相关函数分析法

void ImgProcess::imgAutoCorr()
{
    imwrite("pre_pic.bmp", M); // 转化为灰度图像
    Mat binImg = imread("pre_pic.bmp", IMREAD_GRAYSCALE);
    int deno = 0, nume = 0; // 根据自相关函数公式，分母固定为deno，分子nume会一直变化
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            // 两次循环嵌套，遍历原来的图像，并计算像素值平方和
            deno = deno + (int)binImg.at<uchar>(i, j) * (int)binImg.at<uchar>(i, j);
        }
    }
    // 创建目标图像
    Mat autoCorrImg;
    autoCorrImg.create(M.size(), binImg.type());
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            // 循环遍历新的图像
            for (int m = 0; m < Rows; m++) {
                for (int n = 0; n < Cols; n++) {
                    // 两次循环嵌套得到分子，此处后面的m+i，n+j相当于书上的i+x，j+y
                    if (m + i > Rows - 1 || n + j > Cols - 1) {
                        nume = 0;
                    }
                    else {
                        nume = nume + (int)binImg.at<uchar>(m, n) * (int)binImg.at<uchar>(m + i, n + j); // 分子累加
                    }
                }
            }
            autoCorrImg.at<uchar>(i, j) = saturate_cast<uchar>(nume / deno); // 分子计算完毕，对当前像素赋值
            nume = 0; // 分子归0，等待下次计算
        }
    }
    SaveImage(autoCorrImg, "自相关函数图像.bmp");
}
*/