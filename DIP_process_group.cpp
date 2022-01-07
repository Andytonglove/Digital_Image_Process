#include "DIP_process.h"

/* 下面的函数为图像的频域处理 */
void ImgProcess::ShowSpectrumMap(string name, Complex<double>* pCData, int nTransWidth, int nTransHeight)
{
    // 本函数用以显示灰度图像经变换的频谱图
    double temp = 0, max = abs(pCData[0]), min = abs(pCData[0]);
    double* pCDraw = new double[nTransWidth * nTransHeight];
    for (int i = 0;i < nTransHeight;i++) {
        for (int j = 0;j < nTransWidth;j++) {
            temp = abs(pCData[i * nTransWidth + j]);
            max = (temp > max) ? temp : max;
            min = (temp < min) ? temp : min; // 进行线性伸缩
        }
    }
    for (int i = 0;i < nTransHeight;i++) {
        for (int j = 0;j < nTransWidth;j++) {
            temp = abs(pCData[i * nTransWidth + j]);
            pCDraw[i * nTransWidth + j] = (temp - min) / (max - min) * 25500; // 乘上25500用以频率域明显化！！！
        }
    }
    Mat Mdata;
    Mdata.create(nTransWidth, nTransHeight, CV_8UC1);
    for (int i = 0;i < nTransHeight;i++) {
        for (int j = 0;j < nTransWidth;j++) {
            Mdata.data[j * nTransWidth + i] = pCDraw[j * nTransWidth + i];
        }
    }
    imshow(name, Mdata);
}

void ImgProcess::DFT(Complex<double>* pCTData, int Width, int Height, Complex<double>* pCFData)
{
    /* 傅里叶变换 */
    int x = 0, y = 0; // 循环控制变量
    double temp1, temp2; // 临时变量
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度
    int nXLev, nYLev; // 行列方向上的迭代次数

    // 分别计算傅里叶变换的宽度和高度（2的整数次幂），图像的高度和宽度不一定为2的整数次幂
    temp1 = log(Width) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransWidth = (int)temp2;

    temp1 = log(Height) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransHeight = (int)temp2;
    // 计算x、y方向上的迭代次数
    nXLev = (int)(log(nTransWidth) / log(2) + 0.5);
    nYLev = (int)(log(nTransHeight) / log(2) + 0.5);

    for (y = 0;y < nTransHeight;y++) {
        // x方向上进行快速傅里叶变换
        FFT(&pCTData[nTransWidth * y], &pCFData[nTransWidth * y], nXLev);
    }
    // pcf中存储了pct经过行变换的结果，直接利用一维FFT进行傅里叶行变换（实质相当于是列变换）
    // x方向上进行快速傅里叶变换
    for (y = 0;y < nTransHeight;y++) {

        for (x = 0;x < nTransWidth;x++) {
            pCTData[nTransHeight * x + y] = pCFData[nTransWidth * y + x];
        }
    }
    // 对x方向快速傅里叶变换相当于对原图像进行列方向傅里叶变换
    for (x = 0;x < nTransWidth;x++) {
        FFT(&pCTData[nTransHeight * x], &pCFData[nTransHeight * x], nYLev);
    }

    // pcf中存储有二维傅里叶变换结果，为方便列方向傅里叶变换转置过一次，将pcf转置回来
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            pCTData[nTransWidth * y + x] = pCFData[nTransHeight * x + y];
        }
    }
    memcpy(pCTData, pCFData, sizeof(Complex<double>) * nTransHeight * nTransWidth);
}

void ImgProcess::IDFT(Complex<double>* pCFData, int Width, int Height, Complex<double>* pCTData)
{
    /* 傅里叶逆变换 */
    int x = 0, y = 0; // 循环控制变量
    double temp1, temp2; // 临时变量
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的宽度和高度
    temp1 = log(Width) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransWidth = int(temp2);

    temp1 = log(Height) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransHeight = int(temp2);

    // 分配内存空间
    Complex<double>* pCWork = new Complex<double>[nTransWidth * nTransHeight];
    Complex<double>* pCTemp; // 临时变量

    // 为利用傅里叶正变换，将频率域数据取共轭，然后经正变换直接得到反变换结果的共轭
    for (y = 0; y < nTransHeight; y++){
        for (x = 0;x < nTransWidth;x++){
            pCTemp = &pCFData[nTransWidth * y + x];
            pCWork[nTransWidth * y + x] = Complex<double>(pCTemp->re, -pCTemp->im);
        }
    }
    DFT(pCWork, Width, Height, pCTData); // 调用傅里叶正变换
    // 求得时域点的共轭，求得最终结果，此结果与实际时域值差一个系数
    for (y = 0; y < nTransHeight; y++) {
        for (x = 0; x < nTransWidth; x++) {
            pCTemp = &pCTData[nTransWidth * y + x];
            pCTData[nTransWidth * y + x] = Complex<double>((pCTemp->re / (nTransWidth * nTransHeight)),
                (-pCTemp->im / (nTransWidth * nTransHeight)));
        }
    }
    delete[]pCWork;
    pCWork = NULL;
}

void ImgProcess::FFT(Complex<double>* pCTData, Complex<double>* pCFData, int nLevel)
{
    /* 快速傅里叶变换 */
    // nLevel为傅里叶变换蝶形算法的级数，是2的幂数
    int i = 0, j = 0, k = 0;
    int nCount = (int)pow(2, nLevel); // 傅里叶变换点数
    int nBtFlyLen = 0; // 某一级数长度
    double dAngle; // 变换的角度
    Complex<double>* pCW = new Complex<double>[nCount / 2]; // 存储傅里叶变化所需要的表
    // 计算傅里叶变换系数
    for (i = 0;i < nCount / 2;i++) {
        dAngle = -2 * CV_PI * i / nCount;
        pCW[i] = Complex<double>(cos(dAngle), sin(dAngle));
    }
    // 变换需要的工作空间
    Complex<double>* pCWork1 = new Complex<double>[nCount];
    Complex<double>* pCWork2 = new Complex<double>[nCount];
    Complex<double>* pCTemp;

    memcpy(pCWork1, pCTData, sizeof(Complex<double>) * nCount); // 初始化写入数据
    int nInter = 0; // 临时变量

    // 鲽形算法进行快速傅里叶变换
    for (k = 0;k < nLevel;k++) {
        for (j = 0;j < (int)pow(2, k);j++) {
            nBtFlyLen = (int)pow(2, (nLevel - k)); // 计算长度

            // 倒序重排 加权计算
            for (i = 0;i < nBtFlyLen / 2;i++) {
                nInter = j * nBtFlyLen;
                pCWork2[i + nInter]
                    = pCWork1[i + nInter] + pCWork1[i + nInter + nBtFlyLen / 2];
                pCWork2[i + nInter + nBtFlyLen / 2]
                    = (pCWork1[i + nInter] - pCWork1[i + nInter + nBtFlyLen / 2]) * pCW[(int)(i * pow(2, k))];
            }
        }
        // 交换pcw1和pcw2数据
        pCTemp = pCWork1;
        pCWork1 = pCWork2;
        pCWork2 = pCTemp;
    }
    // ShowSpectrumMap(pCWork1, nTransWidth, nTransHeight);
    // 重新排序
    for (j = 0;j < nCount;j++) {
        nInter = 0;
        for (i = 0;i < nLevel;i++) {
            if (j & (1 << i)) {
                nInter += 1 << (nLevel - i - 1);
            }
        }
        pCFData[j] = pCWork1[nInter];
    }
    // 释放内存空间
    delete[]pCW;
    delete[]pCWork1;
    delete[]pCWork2;
    pCW = NULL;
    pCWork1 = NULL;
    pCWork2 = NULL;
}

void ImgProcess::PreHandle(ImgProcess& img, Complex<double>* &pCTData, Complex<double>* &pCFData,
    int nTransWidth, int nTransHeight)
{
    /* 预处理函数，包括滤波开始前的大部分处理，即一次DFT与中心化等操作 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量

    // 初始化图像的宽和高 补零
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            pCTData[y * nTransWidth + x] = Complex<double>(0, 0);
        }
    }
    // 把数据传给pctdata
    unsigned char unchval; // 图像像素值
    for (y = 0;y < Height;y++) {
        for (x = 0;x < Width;x++) {
            unchval = M.data[y * Width + x]; // *pow(-1, x + y);也可利用傅里叶变换进行中心化
            pCTData[y * nTransWidth + x] = Complex<double>(unchval, 0);
        }
    }
    // ShowSpectrumMap("傅里叶变换前", pCTData, nTransHeight, nTransWidth);

    DFT(pCTData, nTransWidth, nTransHeight, pCFData); // 傅里叶变换！

    // ShowSpectrumMap("傅里叶变换后", pCFData, nTransHeight, nTransWidth);

    // 频域中心化，逆时针ABCD变DBCA
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // 注意：这里易混淆，先进行左右变换，下一个for进行上下变换
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // 序号计数方法=行号*行数+列号依旧不变
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    ShowSpectrumMap("傅里叶变换-频域中心化后", pCFData, nTransHeight, nTransWidth);
}

void ImgProcess::LastHandle(ImgProcess& img, Complex<double>*& pCTData, Complex<double>*& pCFData, 
    int nTransWidth, int nTransHeight, string Mname)
{
    /* 终处理，包括中心化还原，傅里叶反变换和反传数据、输出保存图像 */
    int x = 0, y = 0; // 循环控制变量
    double dReal, dImage; // 实部和虚部值

    // 中心化还原
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // 注意：这里易混淆，先进行左右变换，下一个for进行上下变换
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // 序号计数方法=行号*行数+列号依旧不变
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    // ShowSpectrumMap("中心化还原后频谱", pCFData, nTransHeight, nTransWidth);

    IDFT(pCFData, nTransWidth, nTransHeight, pCTData); // 傅里叶反变换！
    // ShowSpectrumMap("最后图像", pCTData, nTransHeight, nTransWidth);

    // 反变换数据传回给data，注意这里用频域图的宽高
    Mat Mdata;
    Mdata.create(nTransWidth, nTransHeight, CV_8UC1);
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            dReal = pCTData[y * nTransWidth + x].re;
            dImage = pCTData[y * nTransWidth + x].im;
            Mdata.data[y * nTransWidth + x] = (unsigned char)max(0.0, min(255.0, sqrt(dReal * dReal + dImage * dImage)));
        }
    }
    // 释放内存空间
    delete[]pCTData;
    delete[]pCFData;
    pCTData = NULL;
    pCFData = NULL;

    SaveImage(Mdata, Mname); // 以名字保存图片
}

void ImgProcess::ILPF(ImgProcess& img, int nRadius)
{
    /* 理想低通滤波，本函数为滤波全过程函数，后续函数中除滤波外的处理过程使用了预处理函数以便 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数

    int x = 0, y = 0; // 循环控制变量
    double temp1, temp2; // 临时变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度
    double dReal, dImage; // 实部和虚部值

    // 分别计算傅里叶变换的点数（2的整数次幂）
    temp1 = log(Width) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransWidth = (int)temp2;

    temp1 = log(Height) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransHeight = (int)temp2;

    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "您输入的数据过大，请重新输入！";
        return;
    }

    // 初始化图像的宽和高 补零
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            pCTData[y * nTransWidth + x] = Complex<double>(0, 0);
        }
    }
    // 把数据传给pctdata
    unsigned char unchval; // 图像像素值
    for (y = 0;y < Height;y++) {
        for (x = 0;x < Width;x++) {
            unchval = M.data[y * Width + x]; // *pow(-1, x + y);也可利用傅里叶变换进行中心化
            pCTData[y * nTransWidth + x] = Complex<double>(unchval, 0);
        }
    }
    // ShowSpectrumMap("傅里叶变换前", pCTData, nTransHeight, nTransWidth);
    DFT(pCTData, nTransWidth, nTransHeight, pCFData); // 傅里叶变换！
    // ShowSpectrumMap("傅里叶变换后", pCFData, nTransHeight, nTransWidth);

    // 频域中心化，逆时针ABCD变DBCA
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // 注意：这里易混淆，先进行左右变换，下一个for进行上下变换
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // 序号计数方法=行号*行数+列号依旧不变
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    ShowSpectrumMap("傅里叶变换-频域中心化后", pCFData, nTransHeight, nTransWidth);

    // 开始实施理想低通滤波，特别注意这里x、y是左上角原点的坐标数字，而中心化原点应该在中心！
    int Xcenter = 0, Ycenter = 0; // 原点中心化坐标
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // 这里对H(u,v)进行处理
            Xcenter = x - nTransWidth / 2;
            Ycenter = y - nTransHeight / 2;
            if (sqrt(Xcenter * Xcenter + Ycenter * Ycenter) > nRadius) {
                pCFData[y * nTransWidth + x] = Complex<double>(0, 0);
            }
        }
    }

    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);
    // 中心化还原
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // 注意：这里易混淆，先进行左右变换，下一个for进行上下变换
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // 序号计数方法=行号*行数+列号依旧不变
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    // ShowSpectrumMap("中心化还原后频谱", pCFData, nTransHeight, nTransWidth);
    
    IDFT(pCFData, nTransWidth, nTransHeight, pCTData); // 傅里叶反变换！
    // ShowSpectrumMap("最后图像", pCTData, nTransHeight, nTransWidth);

    // 反变换数据传回给data，注意这里用频域图的宽高
    Mat Mdata;
    Mdata.create(nTransWidth, nTransHeight, CV_8UC1);
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            dReal = pCTData[y * nTransWidth + x].re;
            dImage = pCTData[y * nTransWidth + x].im;
            Mdata.data[y * nTransWidth + x] = (unsigned char)max(0.0, min(255.0, sqrt(dReal * dReal + dImage * dImage)));
        }
    }
    // 释放内存空间
    delete[]pCTData;
    delete[]pCFData;
    pCTData = NULL;
    pCFData = NULL;

    SaveImage(Mdata, "理想低通滤波.bmp");
}

void ImgProcess::IHPF(ImgProcess& img, int nRadius)
{
    /* 理想高通滤波。从本函数开始，后置函数均使用预处理和终处理函数 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的点数（2的整数次幂）
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "您输入的数据过大，请重新输入！";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // 开始实施理想高通滤波，特别注意这里x、y是左上角原点的坐标数字，而中心化原点应该在中心！
    int xc = 0, yc = 0; // 原点中心化坐标
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // 这里对H(u,v)进行处理
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            if (sqrt(xc * xc + yc * yc) < nRadius) {
                pCFData[y * nTransWidth + x] = Complex<double>(0, 0);
            }
        }
    }
    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight,"理想高通滤波.bmp");
}

void ImgProcess::BLPF(ImgProcess& img, int nRadius)
{
    /* 巴特沃斯低通滤波 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的点数（2的整数次幂）
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "您输入的数据过大，请重新输入！";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // 下面进行巴特沃斯低通滤波
    int xc = 0, yc = 0;
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            H = (double)(yc * yc + xc * xc);
            H = H / (nRadius * nRadius);
            // H = pow(H, 2 * n);
            H = 1 / (1 + H);
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H);
        }
    }
    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "巴特沃斯低通滤波.bmp");
}

void ImgProcess::BHPF(ImgProcess& img, int nRadius)
{
    /* 巴特沃斯高通滤波 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的点数（2的整数次幂）
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "您输入的数据过大，请重新输入！";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // 下面进行巴特沃斯高通滤波
    int xc = 0, yc = 0;
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            H = (double)(yc * yc + xc * xc);
            H = (nRadius * nRadius) / H;
            // H = pow(H, 2 * n);
            H = 1 / (1 + H);
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H);
        }
    }
    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "巴特沃斯高通滤波.bmp");
}

void ImgProcess::ELPF(ImgProcess& img, int nRadius)
{
    /* 指数低通滤波 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的点数（2的整数次幂）
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "您输入的数据过大，请重新输入！";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // 开始实施指数低通滤波，特别注意这里x、y是左上角原点的坐标数字，而中心化原点应该在中心！
    int xc = 0, yc = 0;
    for (y = 0; y < nTransHeight; y++) {
        for (x = 0; x < nTransWidth; x++) {
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            H = (double)(yc * yc + xc * xc);
            H = H / (nRadius * nRadius);
            H = exp(-H); // 求H值
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H);
        }
    }
    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "指数低通滤波.bmp");
}

void ImgProcess::EHPF(ImgProcess& img, int nRadius)
{
    /* 指数高通滤波 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的点数（2的整数次幂）
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "您输入的数据过大，请重新输入！";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // 开始实施指数高通滤波，特别注意这里x、y是左上角原点的坐标数字，而中心化原点应该在中心！
    int xc = 0, yc = 0;
    for (y = 0; y < nTransHeight; y++) {
        for (x = 0; x < nTransWidth; x++) {
            yc = y - nTransHeight / 2;
            xc = x - nTransWidth / 2;
            H = (double)(yc * yc + xc * xc);
            H = H / (nRadius * nRadius);
            H = 1 - exp(-H); // 求H值
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H);
        }
    }
    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "指数高通滤波.bmp");
}

void ImgProcess::TLPF(ImgProcess& img, int nRadius1, int nRadius2)
{
    /* 梯形低通滤波 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的点数（2的整数次幂）
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2) || nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2) || nRadius2 <= nRadius1) {
        if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2)) {
            cout << "您输入的数据1过大，请重新输入！";
        }
        if (nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2)) {
            cout << "您输入的数据2过大，请重新输入！";
        }
        else {
            cout << "您输入的两个数据顺序相反，请重新输入！";
        }
        return;
    }

    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // 开始实施梯形低通滤波，特别注意这里x、y是左上角原点的坐标数字，而中心化原点应该在中心！
    int xc = 0, yc = 0; // 原点中心化坐标
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // 这里对H(u,v)进行处理
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            H = sqrt((double)(xc * xc + yc * yc));
            if (H < nRadius1) {
                H = 1;
            }
            if ((H >= nRadius1) && (H <= nRadius2)) {
                H = (H - nRadius2) / (nRadius1 - nRadius2);
            }
            if (H > nRadius2) {
                H = 0;
            }
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H); // 卷积
        }
    }
    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "梯形低通滤波图.bmp");
}

void ImgProcess::THPF(ImgProcess& img, int nRadius1, int nRadius2)
{
    /* 梯形高通滤波 */
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    int x = 0, y = 0; // 循环控制变量
    double H = 0; // 滤波系数D0
    int nTransWidth, nTransHeight; // 傅里叶变换的高度和宽度

    // 分别计算傅里叶变换的点数（2的整数次幂）
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 时域指针
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // 频域指针

    if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2) || nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2) || nRadius1 >= nRadius2) {
        if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2)) {
            cout << "您输入的数据1过大，请重新输入！";
        }
        if (nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2)) {
            cout << "您输入的数据2过大，请重新输入！";
        }
        else {
            cout << "您输入的两个数据顺序相反，请重新输入！";
        }
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // 开始实施梯形高通滤波，特别注意这里x、y是左上角原点的坐标数字，而中心化原点应该在中心！
    int xc = 0, yc = 0; // 原点中心化坐标
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // 这里对H(u,v)进行处理
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            H = sqrt((double)(xc * xc + yc * yc)); // R1小R2大
            if (H < nRadius1) {
                H = 0;
            }
            if ((H >= nRadius1) && (H <= nRadius2)) {
                H = (H - nRadius1) / (nRadius2 - nRadius1);
            }
            if (H > nRadius2) {
                H = 1;
            }
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H); // 卷积
        }
    }
    ShowSpectrumMap("滤波后", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "梯形高通滤波图.bmp");
}

void ImgProcess::HomomorphicFilter(ImgProcess& img, double gammaH, double gammaL)
{
    /* 同态滤波，利用cv带的函数实现，主要复现过程 */

    int x = 0, y = 0; // 循环变量 
    int Width = img.M.cols; // 列数
    int Height = img.M.rows; // 行数
    // 分别计算傅里叶变换的点数（2的整数次幂）
    int nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    int nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // 复数时域指针

    // 把数据传给pctdata
    unsigned char unchval; // 图像像素值
    for (y = 0;y < Height;y++) {
        for (x = 0;x < Width;x++) {
            unchval = M.data[y * Width + x];
            pCTData[y * nTransWidth + x] = Complex<double>(unchval, 0);
        }
    }

    Mat Msrc = Mat::ones(nTransHeight, nTransWidth, CV_8UC1); // 新建数据矩阵，用以接受并转换数据
    for (y = 0; y < nTransHeight; y++)
    {
        for (x = 0; x < nTransWidth; x++)
        {
            Msrc.data[y * nTransWidth + x] = pCTData[y * nTransWidth + x].re;
        }
    }
    Msrc.convertTo(Msrc, CV_64FC1); // 矩阵转换
    int Rows = Msrc.rows;
    int Cols = Msrc.cols;
    int m = (Rows % 2 == 1) ? (Rows + 1) : Rows;
    int n = (Cols % 2 == 1) ? (Cols + 1) : Cols;
    copyMakeBorder(Msrc, Msrc, 0, m - Rows, 0, n - Cols, BORDER_CONSTANT, Scalar::all(0)); // 边界处理
    Mat dst(Rows, Cols, CV_64FC1); // 结果图像矩阵

    for (int i = 0; i < Rows; i++) {
        // 对数据数组ln取对数
        double* DataSrc = Msrc.ptr<double>(i);
        double* DataLog = Msrc.ptr<double>(i);
        for (int j = 0; j < Cols; j++) {
            DataLog[j] = log(DataSrc[j] + 0.0001);
        }
    }
    Mat dCT_mat = Mat::zeros(Rows, Cols, CV_64FC1);
    dct(Msrc, dCT_mat); // DCT

    // 高斯同态滤波器
    // double gammaH = 1.5, gammaL = 0.5; // 两个阈值
    double C = 1; // C是参数用以控制低频到高频过渡的陡峭程度
    double d2 = 0;
    double d0 = (Msrc.rows / 2.0) * (Msrc.rows / 2.0) + (Msrc.cols / 2.0) * (Msrc.cols / 2.0);
    Mat H_uv = Mat::zeros(Rows, Cols, CV_64FC1);
    for (int i = 0; i < Rows; i++) {
        double* data_H_uv = H_uv.ptr<double>(i);
        for (int j = 0; j < Cols; j++) {
            // 滤波处理
            d2 = pow(i, 2.0) + pow(j, 2.0);
            data_H_uv[j] = (gammaH - gammaL) * (1 - exp(-C * d2 / d0)) + gammaL;
        }
    }
    H_uv.ptr<double>(0)[0] = 1.1;
    dCT_mat = dCT_mat.mul(H_uv);
    idct(dCT_mat, dst); // IDCT 

    for (int i = 0; i < Rows; i++) {
        // EXP
        double* srcdata = dst.ptr<double>(i);
        double* dstdata = dst.ptr<double>(i);
        for (int j = 0; j < Cols; j++) {
            dstdata[j] = exp(srcdata[j]);
        }
    }
    dst.convertTo(dst, CV_8UC1);
    SaveImage(dst, "同态滤波.bmp"); // 加上后缀名不然易报错
}
