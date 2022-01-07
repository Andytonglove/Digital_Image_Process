#include "DIP_process.h"

/* ����ĺ���Ϊͼ���Ƶ���� */
void ImgProcess::ShowSpectrumMap(string name, Complex<double>* pCData, int nTransWidth, int nTransHeight)
{
    // ������������ʾ�Ҷ�ͼ�񾭱任��Ƶ��ͼ
    double temp = 0, max = abs(pCData[0]), min = abs(pCData[0]);
    double* pCDraw = new double[nTransWidth * nTransHeight];
    for (int i = 0;i < nTransHeight;i++) {
        for (int j = 0;j < nTransWidth;j++) {
            temp = abs(pCData[i * nTransWidth + j]);
            max = (temp > max) ? temp : max;
            min = (temp < min) ? temp : min; // ������������
        }
    }
    for (int i = 0;i < nTransHeight;i++) {
        for (int j = 0;j < nTransWidth;j++) {
            temp = abs(pCData[i * nTransWidth + j]);
            pCDraw[i * nTransWidth + j] = (temp - min) / (max - min) * 25500; // ����25500����Ƶ�������Ի�������
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
    /* ����Ҷ�任 */
    int x = 0, y = 0; // ѭ�����Ʊ���
    double temp1, temp2; // ��ʱ����
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��
    int nXLev, nYLev; // ���з����ϵĵ�������

    // �ֱ���㸵��Ҷ�任�Ŀ�Ⱥ͸߶ȣ�2���������ݣ���ͼ��ĸ߶ȺͿ�Ȳ�һ��Ϊ2����������
    temp1 = log(Width) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransWidth = (int)temp2;

    temp1 = log(Height) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransHeight = (int)temp2;
    // ����x��y�����ϵĵ�������
    nXLev = (int)(log(nTransWidth) / log(2) + 0.5);
    nYLev = (int)(log(nTransHeight) / log(2) + 0.5);

    for (y = 0;y < nTransHeight;y++) {
        // x�����Ͻ��п��ٸ���Ҷ�任
        FFT(&pCTData[nTransWidth * y], &pCFData[nTransWidth * y], nXLev);
    }
    // pcf�д洢��pct�����б任�Ľ����ֱ������һάFFT���и���Ҷ�б任��ʵ���൱�����б任��
    // x�����Ͻ��п��ٸ���Ҷ�任
    for (y = 0;y < nTransHeight;y++) {

        for (x = 0;x < nTransWidth;x++) {
            pCTData[nTransHeight * x + y] = pCFData[nTransWidth * y + x];
        }
    }
    // ��x������ٸ���Ҷ�任�൱�ڶ�ԭͼ������з�����Ҷ�任
    for (x = 0;x < nTransWidth;x++) {
        FFT(&pCTData[nTransHeight * x], &pCFData[nTransHeight * x], nYLev);
    }

    // pcf�д洢�ж�ά����Ҷ�任�����Ϊ�����з�����Ҷ�任ת�ù�һ�Σ���pcfת�û���
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            pCTData[nTransWidth * y + x] = pCFData[nTransHeight * x + y];
        }
    }
    memcpy(pCTData, pCFData, sizeof(Complex<double>) * nTransHeight * nTransWidth);
}

void ImgProcess::IDFT(Complex<double>* pCFData, int Width, int Height, Complex<double>* pCTData)
{
    /* ����Ҷ��任 */
    int x = 0, y = 0; // ѭ�����Ʊ���
    double temp1, temp2; // ��ʱ����
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�Ŀ�Ⱥ͸߶�
    temp1 = log(Width) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransWidth = int(temp2);

    temp1 = log(Height) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransHeight = int(temp2);

    // �����ڴ�ռ�
    Complex<double>* pCWork = new Complex<double>[nTransWidth * nTransHeight];
    Complex<double>* pCTemp; // ��ʱ����

    // Ϊ���ø���Ҷ���任����Ƶ��������ȡ���Ȼ�����任ֱ�ӵõ����任����Ĺ���
    for (y = 0; y < nTransHeight; y++){
        for (x = 0;x < nTransWidth;x++){
            pCTemp = &pCFData[nTransWidth * y + x];
            pCWork[nTransWidth * y + x] = Complex<double>(pCTemp->re, -pCTemp->im);
        }
    }
    DFT(pCWork, Width, Height, pCTData); // ���ø���Ҷ���任
    // ���ʱ���Ĺ��������ս�����˽����ʵ��ʱ��ֵ��һ��ϵ��
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
    /* ���ٸ���Ҷ�任 */
    // nLevelΪ����Ҷ�任�����㷨�ļ�������2������
    int i = 0, j = 0, k = 0;
    int nCount = (int)pow(2, nLevel); // ����Ҷ�任����
    int nBtFlyLen = 0; // ĳһ��������
    double dAngle; // �任�ĽǶ�
    Complex<double>* pCW = new Complex<double>[nCount / 2]; // �洢����Ҷ�仯����Ҫ�ı�
    // ���㸵��Ҷ�任ϵ��
    for (i = 0;i < nCount / 2;i++) {
        dAngle = -2 * CV_PI * i / nCount;
        pCW[i] = Complex<double>(cos(dAngle), sin(dAngle));
    }
    // �任��Ҫ�Ĺ����ռ�
    Complex<double>* pCWork1 = new Complex<double>[nCount];
    Complex<double>* pCWork2 = new Complex<double>[nCount];
    Complex<double>* pCTemp;

    memcpy(pCWork1, pCTData, sizeof(Complex<double>) * nCount); // ��ʼ��д������
    int nInter = 0; // ��ʱ����

    // �����㷨���п��ٸ���Ҷ�任
    for (k = 0;k < nLevel;k++) {
        for (j = 0;j < (int)pow(2, k);j++) {
            nBtFlyLen = (int)pow(2, (nLevel - k)); // ���㳤��

            // �������� ��Ȩ����
            for (i = 0;i < nBtFlyLen / 2;i++) {
                nInter = j * nBtFlyLen;
                pCWork2[i + nInter]
                    = pCWork1[i + nInter] + pCWork1[i + nInter + nBtFlyLen / 2];
                pCWork2[i + nInter + nBtFlyLen / 2]
                    = (pCWork1[i + nInter] - pCWork1[i + nInter + nBtFlyLen / 2]) * pCW[(int)(i * pow(2, k))];
            }
        }
        // ����pcw1��pcw2����
        pCTemp = pCWork1;
        pCWork1 = pCWork2;
        pCWork2 = pCTemp;
    }
    // ShowSpectrumMap(pCWork1, nTransWidth, nTransHeight);
    // ��������
    for (j = 0;j < nCount;j++) {
        nInter = 0;
        for (i = 0;i < nLevel;i++) {
            if (j & (1 << i)) {
                nInter += 1 << (nLevel - i - 1);
            }
        }
        pCFData[j] = pCWork1[nInter];
    }
    // �ͷ��ڴ�ռ�
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
    /* Ԥ�������������˲���ʼǰ�Ĵ󲿷ִ�����һ��DFT�����Ļ��Ȳ��� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���

    // ��ʼ��ͼ��Ŀ�͸� ����
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            pCTData[y * nTransWidth + x] = Complex<double>(0, 0);
        }
    }
    // �����ݴ���pctdata
    unsigned char unchval; // ͼ������ֵ
    for (y = 0;y < Height;y++) {
        for (x = 0;x < Width;x++) {
            unchval = M.data[y * Width + x]; // *pow(-1, x + y);Ҳ�����ø���Ҷ�任�������Ļ�
            pCTData[y * nTransWidth + x] = Complex<double>(unchval, 0);
        }
    }
    // ShowSpectrumMap("����Ҷ�任ǰ", pCTData, nTransHeight, nTransWidth);

    DFT(pCTData, nTransWidth, nTransHeight, pCFData); // ����Ҷ�任��

    // ShowSpectrumMap("����Ҷ�任��", pCFData, nTransHeight, nTransWidth);

    // Ƶ�����Ļ�����ʱ��ABCD��DBCA
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // ע�⣺�����׻������Ƚ������ұ任����һ��for�������±任
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // ��ż�������=�к�*����+�к����ɲ���
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    ShowSpectrumMap("����Ҷ�任-Ƶ�����Ļ���", pCFData, nTransHeight, nTransWidth);
}

void ImgProcess::LastHandle(ImgProcess& img, Complex<double>*& pCTData, Complex<double>*& pCFData, 
    int nTransWidth, int nTransHeight, string Mname)
{
    /* �մ����������Ļ���ԭ������Ҷ���任�ͷ������ݡ��������ͼ�� */
    int x = 0, y = 0; // ѭ�����Ʊ���
    double dReal, dImage; // ʵ�����鲿ֵ

    // ���Ļ���ԭ
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // ע�⣺�����׻������Ƚ������ұ任����һ��for�������±任
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // ��ż�������=�к�*����+�к����ɲ���
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    // ShowSpectrumMap("���Ļ���ԭ��Ƶ��", pCFData, nTransHeight, nTransWidth);

    IDFT(pCFData, nTransWidth, nTransHeight, pCTData); // ����Ҷ���任��
    // ShowSpectrumMap("���ͼ��", pCTData, nTransHeight, nTransWidth);

    // ���任���ݴ��ظ�data��ע��������Ƶ��ͼ�Ŀ��
    Mat Mdata;
    Mdata.create(nTransWidth, nTransHeight, CV_8UC1);
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            dReal = pCTData[y * nTransWidth + x].re;
            dImage = pCTData[y * nTransWidth + x].im;
            Mdata.data[y * nTransWidth + x] = (unsigned char)max(0.0, min(255.0, sqrt(dReal * dReal + dImage * dImage)));
        }
    }
    // �ͷ��ڴ�ռ�
    delete[]pCTData;
    delete[]pCFData;
    pCTData = NULL;
    pCFData = NULL;

    SaveImage(Mdata, Mname); // �����ֱ���ͼƬ
}

void ImgProcess::ILPF(ImgProcess& img, int nRadius)
{
    /* �����ͨ�˲���������Ϊ�˲�ȫ���̺��������������г��˲���Ĵ������ʹ����Ԥ�������Ա� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����

    int x = 0, y = 0; // ѭ�����Ʊ���
    double temp1, temp2; // ��ʱ����
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��
    double dReal, dImage; // ʵ�����鲿ֵ

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    temp1 = log(Width) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransWidth = (int)temp2;

    temp1 = log(Height) / log(2);
    temp2 = ceil(temp1);
    temp2 = pow(2, temp2);
    nTransHeight = (int)temp2;

    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "����������ݹ������������룡";
        return;
    }

    // ��ʼ��ͼ��Ŀ�͸� ����
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            pCTData[y * nTransWidth + x] = Complex<double>(0, 0);
        }
    }
    // �����ݴ���pctdata
    unsigned char unchval; // ͼ������ֵ
    for (y = 0;y < Height;y++) {
        for (x = 0;x < Width;x++) {
            unchval = M.data[y * Width + x]; // *pow(-1, x + y);Ҳ�����ø���Ҷ�任�������Ļ�
            pCTData[y * nTransWidth + x] = Complex<double>(unchval, 0);
        }
    }
    // ShowSpectrumMap("����Ҷ�任ǰ", pCTData, nTransHeight, nTransWidth);
    DFT(pCTData, nTransWidth, nTransHeight, pCFData); // ����Ҷ�任��
    // ShowSpectrumMap("����Ҷ�任��", pCFData, nTransHeight, nTransWidth);

    // Ƶ�����Ļ�����ʱ��ABCD��DBCA
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // ע�⣺�����׻������Ƚ������ұ任����һ��for�������±任
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // ��ż�������=�к�*����+�к����ɲ���
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    ShowSpectrumMap("����Ҷ�任-Ƶ�����Ļ���", pCFData, nTransHeight, nTransWidth);

    // ��ʼʵʩ�����ͨ�˲����ر�ע������x��y�����Ͻ�ԭ����������֣������Ļ�ԭ��Ӧ�������ģ�
    int Xcenter = 0, Ycenter = 0; // ԭ�����Ļ�����
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // �����H(u,v)���д���
            Xcenter = x - nTransWidth / 2;
            Ycenter = y - nTransHeight / 2;
            if (sqrt(Xcenter * Xcenter + Ycenter * Ycenter) > nRadius) {
                pCFData[y * nTransWidth + x] = Complex<double>(0, 0);
            }
        }
    }

    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);
    // ���Ļ���ԭ
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth / 2;x++) {
            // ע�⣺�����׻������Ƚ������ұ任����һ��for�������±任
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[y * nTransWidth + x + nTransWidth / 2];
            pCFData[y * nTransWidth + x + nTransWidth / 2] = tmp;
        }
    }
    for (x = 0;x < nTransWidth;x++) {
        for (y = 0;y < nTransHeight / 2;y++) {
            // ��ż�������=�к�*����+�к����ɲ���
            Complex<double> tmp = pCFData[y * nTransWidth + x];
            pCFData[y * nTransWidth + x] = pCFData[(y + nTransHeight / 2) * nTransWidth + x];
            pCFData[(y + nTransHeight / 2) * nTransWidth + x] = tmp;
        }
    }
    // ShowSpectrumMap("���Ļ���ԭ��Ƶ��", pCFData, nTransHeight, nTransWidth);
    
    IDFT(pCFData, nTransWidth, nTransHeight, pCTData); // ����Ҷ���任��
    // ShowSpectrumMap("���ͼ��", pCTData, nTransHeight, nTransWidth);

    // ���任���ݴ��ظ�data��ע��������Ƶ��ͼ�Ŀ��
    Mat Mdata;
    Mdata.create(nTransWidth, nTransHeight, CV_8UC1);
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            dReal = pCTData[y * nTransWidth + x].re;
            dImage = pCTData[y * nTransWidth + x].im;
            Mdata.data[y * nTransWidth + x] = (unsigned char)max(0.0, min(255.0, sqrt(dReal * dReal + dImage * dImage)));
        }
    }
    // �ͷ��ڴ�ռ�
    delete[]pCTData;
    delete[]pCFData;
    pCTData = NULL;
    pCFData = NULL;

    SaveImage(Mdata, "�����ͨ�˲�.bmp");
}

void ImgProcess::IHPF(ImgProcess& img, int nRadius)
{
    /* �����ͨ�˲����ӱ�������ʼ�����ú�����ʹ��Ԥ������մ����� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "����������ݹ������������룡";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // ��ʼʵʩ�����ͨ�˲����ر�ע������x��y�����Ͻ�ԭ����������֣������Ļ�ԭ��Ӧ�������ģ�
    int xc = 0, yc = 0; // ԭ�����Ļ�����
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // �����H(u,v)���д���
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            if (sqrt(xc * xc + yc * yc) < nRadius) {
                pCFData[y * nTransWidth + x] = Complex<double>(0, 0);
            }
        }
    }
    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight,"�����ͨ�˲�.bmp");
}

void ImgProcess::BLPF(ImgProcess& img, int nRadius)
{
    /* ������˹��ͨ�˲� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "����������ݹ������������룡";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // ������а�����˹��ͨ�˲�
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
    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "������˹��ͨ�˲�.bmp");
}

void ImgProcess::BHPF(ImgProcess& img, int nRadius)
{
    /* ������˹��ͨ�˲� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "����������ݹ������������룡";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // ������а�����˹��ͨ�˲�
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
    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "������˹��ͨ�˲�.bmp");
}

void ImgProcess::ELPF(ImgProcess& img, int nRadius)
{
    /* ָ����ͨ�˲� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "����������ݹ������������룡";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // ��ʼʵʩָ����ͨ�˲����ر�ע������x��y�����Ͻ�ԭ����������֣������Ļ�ԭ��Ӧ�������ģ�
    int xc = 0, yc = 0;
    for (y = 0; y < nTransHeight; y++) {
        for (x = 0; x < nTransWidth; x++) {
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            H = (double)(yc * yc + xc * xc);
            H = H / (nRadius * nRadius);
            H = exp(-H); // ��Hֵ
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H);
        }
    }
    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "ָ����ͨ�˲�.bmp");
}

void ImgProcess::EHPF(ImgProcess& img, int nRadius)
{
    /* ָ����ͨ�˲� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius > (nTransHeight / 2) || nRadius > (nTransWidth / 2)) {
        cout << "����������ݹ������������룡";
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // ��ʼʵʩָ����ͨ�˲����ر�ע������x��y�����Ͻ�ԭ����������֣������Ļ�ԭ��Ӧ�������ģ�
    int xc = 0, yc = 0;
    for (y = 0; y < nTransHeight; y++) {
        for (x = 0; x < nTransWidth; x++) {
            yc = y - nTransHeight / 2;
            xc = x - nTransWidth / 2;
            H = (double)(yc * yc + xc * xc);
            H = H / (nRadius * nRadius);
            H = 1 - exp(-H); // ��Hֵ
            pCFData[y * nTransWidth + x] = Complex<double>(pCFData[y * nTransWidth + x].re * H,
                pCFData[y * nTransWidth + x].im * H);
        }
    }
    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "ָ����ͨ�˲�.bmp");
}

void ImgProcess::TLPF(ImgProcess& img, int nRadius1, int nRadius2)
{
    /* ���ε�ͨ�˲� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2) || nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2) || nRadius2 <= nRadius1) {
        if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2)) {
            cout << "�����������1�������������룡";
        }
        if (nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2)) {
            cout << "�����������2�������������룡";
        }
        else {
            cout << "���������������˳���෴�����������룡";
        }
        return;
    }

    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // ��ʼʵʩ���ε�ͨ�˲����ر�ע������x��y�����Ͻ�ԭ����������֣������Ļ�ԭ��Ӧ�������ģ�
    int xc = 0, yc = 0; // ԭ�����Ļ�����
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // �����H(u,v)���д���
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
                pCFData[y * nTransWidth + x].im * H); // ���
        }
    }
    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "���ε�ͨ�˲�ͼ.bmp");
}

void ImgProcess::THPF(ImgProcess& img, int nRadius1, int nRadius2)
{
    /* ���θ�ͨ�˲� */
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    int x = 0, y = 0; // ѭ�����Ʊ���
    double H = 0; // �˲�ϵ��D0
    int nTransWidth, nTransHeight; // ����Ҷ�任�ĸ߶ȺͿ��

    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ʱ��ָ��
    Complex<double>* pCFData = new Complex<double>[nTransWidth * nTransHeight]; // Ƶ��ָ��

    if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2) || nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2) || nRadius1 >= nRadius2) {
        if (nRadius1 > (nTransHeight / 2) || nRadius1 > (nTransWidth / 2)) {
            cout << "�����������1�������������룡";
        }
        if (nRadius2 > (nTransHeight / 2) || nRadius2 > (nTransWidth / 2)) {
            cout << "�����������2�������������룡";
        }
        else {
            cout << "���������������˳���෴�����������룡";
        }
        return;
    }
    PreHandle(img, pCTData, pCFData, nTransWidth, nTransHeight);

    // ��ʼʵʩ���θ�ͨ�˲����ر�ע������x��y�����Ͻ�ԭ����������֣������Ļ�ԭ��Ӧ�������ģ�
    int xc = 0, yc = 0; // ԭ�����Ļ�����
    for (y = 0;y < nTransHeight;y++) {
        for (x = 0;x < nTransWidth;x++) {
            // �����H(u,v)���д���
            xc = x - nTransWidth / 2;
            yc = y - nTransHeight / 2;
            H = sqrt((double)(xc * xc + yc * yc)); // R1СR2��
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
                pCFData[y * nTransWidth + x].im * H); // ���
        }
    }
    ShowSpectrumMap("�˲���", pCFData, nTransHeight, nTransWidth);

    LastHandle(img, pCTData, pCFData, nTransWidth, nTransHeight, "���θ�ͨ�˲�ͼ.bmp");
}

void ImgProcess::HomomorphicFilter(ImgProcess& img, double gammaH, double gammaL)
{
    /* ̬ͬ�˲�������cv���ĺ���ʵ�֣���Ҫ���ֹ��� */

    int x = 0, y = 0; // ѭ������ 
    int Width = img.M.cols; // ����
    int Height = img.M.rows; // ����
    // �ֱ���㸵��Ҷ�任�ĵ�����2���������ݣ�
    int nTransWidth = (int)pow(2, ceil(log(Width) / log(2)));
    int nTransHeight = (int)pow(2, ceil(log(Height) / log(2)));
    Complex<double>* pCTData = new Complex<double>[nTransWidth * nTransHeight]; // ����ʱ��ָ��

    // �����ݴ���pctdata
    unsigned char unchval; // ͼ������ֵ
    for (y = 0;y < Height;y++) {
        for (x = 0;x < Width;x++) {
            unchval = M.data[y * Width + x];
            pCTData[y * nTransWidth + x] = Complex<double>(unchval, 0);
        }
    }

    Mat Msrc = Mat::ones(nTransHeight, nTransWidth, CV_8UC1); // �½����ݾ������Խ��ܲ�ת������
    for (y = 0; y < nTransHeight; y++)
    {
        for (x = 0; x < nTransWidth; x++)
        {
            Msrc.data[y * nTransWidth + x] = pCTData[y * nTransWidth + x].re;
        }
    }
    Msrc.convertTo(Msrc, CV_64FC1); // ����ת��
    int Rows = Msrc.rows;
    int Cols = Msrc.cols;
    int m = (Rows % 2 == 1) ? (Rows + 1) : Rows;
    int n = (Cols % 2 == 1) ? (Cols + 1) : Cols;
    copyMakeBorder(Msrc, Msrc, 0, m - Rows, 0, n - Cols, BORDER_CONSTANT, Scalar::all(0)); // �߽紦��
    Mat dst(Rows, Cols, CV_64FC1); // ���ͼ�����

    for (int i = 0; i < Rows; i++) {
        // ����������lnȡ����
        double* DataSrc = Msrc.ptr<double>(i);
        double* DataLog = Msrc.ptr<double>(i);
        for (int j = 0; j < Cols; j++) {
            DataLog[j] = log(DataSrc[j] + 0.0001);
        }
    }
    Mat dCT_mat = Mat::zeros(Rows, Cols, CV_64FC1);
    dct(Msrc, dCT_mat); // DCT

    // ��˹̬ͬ�˲���
    // double gammaH = 1.5, gammaL = 0.5; // ������ֵ
    double C = 1; // C�ǲ������Կ��Ƶ�Ƶ����Ƶ���ɵĶ��ͳ̶�
    double d2 = 0;
    double d0 = (Msrc.rows / 2.0) * (Msrc.rows / 2.0) + (Msrc.cols / 2.0) * (Msrc.cols / 2.0);
    Mat H_uv = Mat::zeros(Rows, Cols, CV_64FC1);
    for (int i = 0; i < Rows; i++) {
        double* data_H_uv = H_uv.ptr<double>(i);
        for (int j = 0; j < Cols; j++) {
            // �˲�����
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
    SaveImage(dst, "̬ͬ�˲�.bmp"); // ���Ϻ�׺����Ȼ�ױ���
}
