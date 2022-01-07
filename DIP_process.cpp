#include "DIP_process.h"

void ImgProcess::SaveImage(Mat &imgM, string imageName)
{
    /* ���������ͼ��ı��漰��ʾ */
    if (imgM.empty()) {
        cout << "ͼ��������⣬�޷���ȷ���棡"; // �������
        waitKey(6000);
    }
    namedWindow("image", 0); // 1Ϊ��ͼƬ��С��ʾ��0Ϊ���ݴ��ڴ�С����
    imshow("image", imgM); // ��ʾͼƬ
    waitKey(); // �����waitkey�Ǳ�Ҫ�ģ���Ȼ����������
    imwrite(imageName, imgM); // ����ͼƬ
}

ImgProcess::ImgProcess(string imageName)
{
    /* ���캯������ͼ�񲢽��г������� */
    M = imread(imageName); // ����ͼƬ���ɳ��������ж���color����grayscale

    // �ж��ļ��Ƿ������򿪣�
    if (M.empty())
    {
        cout << "�޷���ͼ�������ļ�·����\n";
        return;
    }
    cout << "ͼ��򿪳ɹ������Խ�����һ�����������\n";

    //��ʼ���������б���
    this->Cols = M.cols;
    this->Rows = M.rows;
    this->bands = M.channels();
    this->pixels = Cols * Rows * bands;
    this->ptr = M.data;
}

ImgProcess::ImgProcess(string imgName,int type)
{
    /* ���캯����ָ����ȡ�ļ����͵Ĺ��캯�� */
    M = imread(imgName, type); // ����ͼƬ���ɳ��������ж���color����grayscale

    // �ж��ļ��Ƿ������򿪣�
    if (M.empty())
    {
        cout << "�޷���ͼ�������ļ�·����\n";
        return;
    }
    cout << "ͼ��򿪳ɹ������Խ�����һ�����������\n";

    //��ʼ���������б���
    this->Cols = M.cols;
    this->Rows = M.rows;
    this->bands = M.channels();
    this->pixels = Cols * Rows * bands;
    this->ptr = M.data;
}

/* ����ĺ�����ͼ����е����㣺�Ҷ����Ա任 */
void ImgProcess::grayLinearTransfer(double k, double b)
{
    /* ���ﱾ������ͼ����лҶ����Ա任 */
    Mat grayImg = Mat::zeros(M.size(), M.type()); // ��ʼ��ͼ���С

    //�Ҷȱ任
    for (int r = 0; r < Rows; r++) {
        for (int c = 0; c < Cols; c++) {
            if (M.channels() == 3) {
                // Vec3b����ʾ�Բ�ɫͼ����д���������ɫ�ǰ���b��g��r���������
                // saturate_cast ����ָ���������͵��������֤��ͼ���ڽ��лҶ����Ա任�󲻻����ɫ�����
                // �������ֵ��������0-256
                grayImg.at<Vec3b>(r, c)[0] = saturate_cast<uchar>((k * M.at<Vec3b>(r, c)[0] + b)); // b
                grayImg.at<Vec3b>(r, c)[1] = saturate_cast<uchar>((k * M.at<Vec3b>(r, c)[1] + b)); // g
                grayImg.at<Vec3b>(r, c)[2] = saturate_cast<uchar>((k * M.at<Vec3b>(r, c)[2] + b)); // r
            }else if (M.channels() == 1) {
                // uchar�ԻҶ�ͼ����д���
                int v = M.at<uchar>(r, c);
                grayImg.at<uchar>(r, c) = saturate_cast<uchar>(k * v + b);
            }
            else {
                cout << "��֧�ֱ�������ͼ��ĻҶȱ任��" << endl;
            }
        }
    }

    /*���ú�������ͼƬ���ر�ע��opencv����ֻ���ر��趨����Ϊλͼ��ʽ������ᱨ�������*/
    SaveImage(grayImg, "�Ҷ����Ա任ͼ��.bmp");
}

/* �����2��������ͼ����оֲ�������ͨ/��ͨ����ֵ�˲������� */
void ImgProcess::PassFilters(int type)
{
    /* ��������ͼ����и����˲����� */
    Mat passImg = Mat::zeros(M.size(), M.type());
    unsigned char* pDst = passImg.data;
    vector<double> _template; // ��ʼ��
    string args = "";

    /* �˲�ѡ�� */
    switch (type)
    {
    case 1:
        _template = highPTemplate1;
        args = "��ͨ�˲�1.bmp";
        break;
    case 2:
        _template = highPTemplate2;
        args = "��ͨ�˲�2.bmp";
        break;
    case 3:
        _template = highPTemplate3;
        args = "������˹��ͨ�˲�3.bmp";
        break;
    case 4:
        _template = lowPTemplate;
        args = "��ͨ�˲�.bmp";
        break;
    default:
        /*��ֵ�˲���Ҫ������ķ�ʽ���ô���*/
        cout << "û�д����˲��������˳���";
        break;
    }

    int tempSize = sqrt(_template.size()); // ��ȡģ���ܴ�С
    int directSize = (tempSize - 1) / 2; // ���ڼ����ģ��ķ�����չ��Χ
    vector<int> myPixels; // Ϊģ���漰�����ڴ濪��һ��vector���飬ʹ��vector���������ڴ��������
    int pixelNum = 0; // ��ǰ������ǵ�n������
    int totalpixel = 0; // �����ܻҶȼ�

    // ���ȣ�����nͨ��������㷶ΧΪ�С����г���С�ڵ�����չ��Χ�ı�Ե����
    for (int i = 0; i < bands; i++) {
        for (int j = directSize; j < Rows - directSize; j++) {
            for (int k = directSize; k < Cols - directSize; k++) {
                // ������
                for (int m = j - directSize; m <= j + directSize; m++) {
                    for (int n = k - directSize; n <= k + directSize; n++) {
                        // vector����ĩβ���Ͻ���Ԫ�أ���ģ��ĵ�size��Ԫ�س��Զ�Ӧλ�õ�����ֵ
                        myPixels.push_back(_template[myPixels.size()] * ptr[(m * Cols + n) * bands + i]);
                    }
                }
                totalpixel = accumulate(myPixels.begin(), myPixels.end(), 0);// �ۼӸ�������
                myPixels.clear(); // �����ʱ����������
                // ���ƻҶȼ�ʹ֮�������ʹ�����Χ��ԭ��Χ��
                pDst[(j * Cols + k) * bands + i] = saturate_cast<uchar>(totalpixel);
            }
        }
    }
    SaveImage(passImg, args);
}

void ImgProcess::midPassFilter()
{
    /* ��������ͼ�������ֵ�˲�������Ĭ��3*3Ϊ�� */
    Mat passImg = Mat::zeros(M.size(), M.type());
    unsigned char* pNtr = passImg.data;
    int const num = 3 * 3;
    uchar pixel[num] = { 0 }; // �������������ֵ

    //��������ĵ㣬3*3��ȥ�������еĵ���Ҫƫ�Ƶ�λ�ã��൱��һ��λ��ƫ�ƾ���
    int delta[3 * 3][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1} };

    for (int k = 0; k < bands; k++) {
        for (int i = 1; i < Rows - 1; i++) {
            for (int j = 1; j < Cols - 1; j++) {
                // ��ȡ3*3����ֵ
                for (int c = 0; c < num; ++c)
                {
                    pixel[c] = ptr[((i + delta[c][0]) * Cols + (j + delta[c][1])) * bands + k];
                }
                sort(pixel, pixel + 9); // ����sort���������ҵ��м�ֵ
                pNtr[(i * Cols + j) * bands + k] = pixel[5];
            }
        }
    }
    SaveImage(passImg, "��ֵ�˲�.bmp");
}

/* �����2��������ͼ����ж�ֵ������ */
void ImgProcess::imgStateBin()
{
    /* ״̬������ȷ��� */
    int T = 0; // ��ֵ
    int nNewThre = 0;
    //��ʼ��������1����2������������1����2�ҶȾ�ֵ
    int cnt1 = 0, cnt2 = 0, mval1 = 0, mval2 = 0;
    int iter = 0; // ��������
    // �Ƚ���ɫͼ��Ϊ�Ҷ�ͼ��
    imwrite("pic_prebin.bmp", M);
    Mat binImg = imread("pic_prebin.bmp", IMREAD_GRAYSCALE);
    DrawgrayHist(M);

    int nEISize = binImg.elemSize(); // ��ȡÿ�����ص��ֽ���
    int G = pow(2, double(8 * nEISize)); // �Ҷȼ���
    nNewThre = int(G / 2); // ����ֵ��������ֵ
    // ����Ҷȼ����������ڴ棬���沢����Ҷ�ֱ��ͼ
    auto* hist = new int[G]; // �Ҷ�ͳ������
    for (int i = 0; i < G; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            int g = binImg.at<uchar>(i, j);
            hist[g]++;
        }
    }
    // �����������ֵ
    for (iter = 0;iter < 100;iter++) {
        /* ��ʼ�� */
        T = nNewThre;
        for (int m = T; m < G; m++) {
            cnt2 += hist[m];
        }
        cnt1 = Cols * Rows - cnt2;
        // ��1��0��ʼ������С��i����2���������G�ݼ�������i
        for (int n = T; n < G; n++) {
            mval2 += (double(hist[n]) / cnt2) * n;
        }
        for (int k = 0; k < T; k++) {
            mval1 += (double(hist[k]) / cnt1) * k;
        }
        T = int(mval1 + mval2) / 2; //������ֵ
    }
    for (int i = 0;i < Rows;i++) {
        for (int j = 0;j < Cols;j++) {
            binImg.at<uchar>(i, j) = (binImg.at<uchar>(i, j) > T) ? (G - 1) : 0; // �ر�ע�������������ֵ��G-1������
        }
    }
    SaveImage(binImg, "״̬����ֵ��.bmp");

    delete[]hist;
}

void ImgProcess::imgAnalysisBin()
{
    /* �жϷ�����������ԭ��ʹ����ֵ���ֵ�����Ҷȼ�֮�䣬������䷽������ */
    /* һЩֵ�Ķ������ʼ�� */
    double ratio = 0;
    int Thre = 0; // ��ֵ
    double cnt1 = 0, cnt2 = 0; // �������ص�������
    double mval1 = 0, mval2 = 0, mval = 0; // �������صĻҶ�ƽ��ֵ�Լ�����ͼ��ĻҶ�ƽ��ֵ
    double delta1 = 0, delta2 = 0; // ��һ�顢�ڶ������صķ���
    double deltaW = 0, deltaB = 0; // ���ڷ����䷽��

    // �Ƚ���ɫͼ��Ϊ�Ҷ�ͼ��
    imwrite("pic_prebin.bmp", M);
    Mat binImg = imread("pic_prebin.bmp", IMREAD_GRAYSCALE);
    DrawgrayHist(M);

    int nEISize = binImg.elemSize(); // ��ȡÿ�����ص��ֽ���
    int G = pow(2, double(8 * nEISize)); // �Ҷȼ���
    // ����Ҷȼ����������ڴ棬���沢����Ҷ�ֱ��ͼ
    auto* hist = new int[G]; // �Ҷ�ͳ������
    for (int i = 0; i < G; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            int H = binImg.at<uchar>(i, j);
            hist[H]++;
        }
    }

    // ͨ��ʹ���ڷ�������䷽��֮�������ȷ����ֵT
    for (int i = 0; i < G; i++) {
        for (int m = i; m < G; m++) {
            cnt2 += hist[m];
        }
        cnt1 = Cols * Rows - cnt2;

        // ��1��0��ʼ������С��i����2���������G�ݼ�������i
        for (int n = i; n < G; n++) {
            mval2 += (double(hist[n]) / cnt2) * n;
        }
        for (int k = 0; k < i; k++) {
            mval1 += (double(hist[k]) / cnt1) * k;
        }
        
        // ����ͼ��ĻҶ�ƽ��ֵ����
        mval = (mval1 * cnt1 + mval2 * cnt2) / (cnt1 + cnt2);

        // ����ķ����Լ����ڷ������䷽����㣬ͬ����
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
            Thre = i; // ��ֵT����
        }
        // ���¸�ֵ��ԭΪ0
        cnt1 = 0;
        cnt2 = 0;
        mval1 = 0;
        mval2 = 0;
        delta1 = 0;
        delta2 = 0;
        deltaW = 0;
        deltaB = 0;
    }

    // ������ֵ���ж�ֵ������
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            binImg.at<uchar>(i, j) = (binImg.at<uchar>(i, j) < Thre) ? 0 : (G - 1); // ͨ����ֵ����ֵ
        }
    }
    SaveImage(binImg, "�жϷ�������ֵ��.bmp");

    delete[]hist;
}

/* �����2������Ϊֱ��ͼƥ�� */
Mat ImgProcess::equalize_hist(Mat& M)
{
    /* ����������ֱ��ͼ���⻯�����ؾ��⻯��ľ��󣬱�����һ������ */
    Mat OutputM = M.clone(); // ���ص����ͼ�����
    int gray[256] = { 0 }; // ��¼ÿ���Ҷȼ����µ����ظ���
    double gray_prob[256] = { 0 }; // ��¼�Ҷȷֲ��ܶ�
    double gray_distribution[256] = { 0 }; // ��¼�ۼ��ܶ�
    int gray_equal[256] = { 0 }; // ���⻯��ĻҶ�ֵ
    int gray_sum = M.cols * M.rows; // ��������

    // ͳ��ÿ���Ҷ��µ����ظ���
    for (int i = 0; i < M.rows; i++)
    {
        uchar* ptr = M.ptr<uchar>(i);
        for (int j = 0; j < M.cols; j++)
        {
            int vaule = ptr[j];
            gray[vaule]++;
        }
    }

    // ͳ�ƻҶ�Ƶ�ʺ��ۼ��ܶ�
    for (int i = 0; i < 256; i++)
    {
        gray_prob[i] = ((double)gray[i] / gray_sum);
    }
    gray_distribution[0] = gray_prob[0];
    for (int i = 1; i < 256; i++)
    {
        gray_distribution[i] = gray_distribution[i - 1] + gray_prob[i];
    }

    // ���¼�����⻯��ĻҶ�ֵ��(N-1)*T+0.5����������
    for (int i = 0; i < 256; i++)
    {
        gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
    }

    // ֱ��ͼ���⻯,����ԭͼÿ���������ֵ
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
    /* ���������Ի��ƻҶ�ֱ��ͼ */
    // �������ֱ��ͼ��������Ϊ�˷�����ʹ���˲���cv����
    int channels = 0; // ͨ����
    MatND dstHist; // ֱ��ͼ����Ľ���洢�ռ䣬��MatND�������洢���  
    int histSize[] = { 256 }; // ֱ��ͼ��ÿһ��ά�ȵ�������
    float midRanges[] = { 0, 256 }; // ȡֵ��Χ
    const float* ranges[] = { midRanges };
    calcHist(&M, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);
    // ps ����Ϊ�˷��㣬����cv�ĺ���������ʵ�ֻҶ�ֱ��ͼ���㣬�����ú���ʵ�ֵĺ��Ĵ����Ѿ������������

    Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3); // ����һ��8λ��3ͨ��ͼ��
    // ͬ�ϣ���cv2�������õ�ֱ��ͼͳ�����������������з�Χ����
    double g_dHistMaxValue;
    minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);
    int size = 256; // ֱ��ͼ�ߴ�
    //����ֱ��ͼ�õ������ݻ�ͼ
    for (int i = 0; i < 256; i++)
    {
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);
        rectangle(drawImage, Point(i, size - 1),
            Point(i, size - value), Scalar(255, 255, 255)); // ����ֱ��ͼ
    }
    imshow("�Ҷ�ֱ��ͼͼ��", drawImage);
    waitKey(); // �������waitkey����Ȼ���������֪ԭ��
}

void ImgProcess::histogramMatching(ImgProcess &matcher)
{
    /* ����������ֱ��ͼ�涨�� */
    Mat image_matched = Mat::zeros(M.rows, M.cols, M.type());
    Mat lu_table(1, 256, CV_8U); // opencv�ṩ��LUT��ӳ�䷽���������Լ�ʵ��һ�����׵�
    int* _gray = new int[256]; // ��Ҫ���淶��ͼ���ԭֱ��ͼ
    int* gray = new int[256]; // ���Թ涨ֱ��ͼͼ���ֱ��ͼ
    for (int i = 0;i < 256;i++) {
        _gray[i] = 0;
        gray[i] = 0;
    }

    // ����ǲ�ɫͼ����ô��ת��Ϊ�Ҷ�ͼ��
    imwrite("pre_pic_this.bmp", this->M); // ����Ǳ��淶���ģ�Ҫ��_gray[]��������
    Mat ImgM = imread("pre_pic_this.bmp", IMREAD_GRAYSCALE);
    imwrite("pre_pic_matcher.bmp", matcher.M);
    Mat ImgThis = imread("pre_pic_matcher.bmp", IMREAD_GRAYSCALE);
    image_matched = ImgM.clone();

    // ������һ���������ֱ��ͼ���⻯
    equalize_hist(ImgM);
    equalize_hist(ImgThis);

    // ����ѭ���ֱ������⻯��Ҷ�����
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
    // ��ø��ʲ�ֵ�������ҵ���ӽ��ĵ㣬������ƥ��
    float(*differ)[256] = new float[256][256];// ֱ�ӷ���ջ�����������Ҫ�����ö�
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++)
        {
            differ[i][j] = fabs(gray[i] - _gray[j]); // ˭ǰ˭������ν
        }
    }
    for (int i = 0; i < 256; i++)
    {
        //��ǰ�������Ҷ�����ӽ�����С��ֵ
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
    // ͼ����ʵ��ӳ��
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
    SaveImage(image_matched, "ֱ��ͼƥ��.bmp");
    DrawgrayHist(image_matched); // ���ú������ƻҶ�ֱ��ͼ

    delete[]gray;
    delete[]_gray;
    delete[]differ;
}

/* �����2������Ϊɫ��ƽ�� */
void ImgProcess::whiteBalance()
{
    /* ��ƽ�⴦�� */
    Mat dst;
    dst.create(M.size(), M.type());
    int HistRGB[767] = { 0 }; // ���飬��ͨ��256*3=768��
    int MaxVal = 0; // RGB��ֵ
    int Threshold = 0; // ��ֵ
    int sum = 0;
    float ratio = 0.1; // ����
    int B_avg = 0, G_avg = 0, R_avg = 0, count = 0; // ������ɫ�;�ֵ��

    // ͳ�Ƶõ�R��G��B�е����ֵMaxVal����ɫ��
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
    // ������ɫ�͵���������������������������ֵ���������ֵThreshold
    for (int i = 766; i >= 0; i--) {
        sum += HistRGB[i];
        if (sum > M.rows * M.cols * ratio) {
            Threshold = i;
            break;
        }
    }
    // ������ɫ�ʹ�����ֵ�����е�ľ�ֵ
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
    //�õ���ֵ
    int color[3] = { 0 }; // 0��1��2�ֱ��Ӧ���̺�
    int avg[3] = { B_avg/count, G_avg/count, R_avg/count }; // �õ���ֵ
    
    // ���·���õ��¾���dst
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
    SaveImage(dst, "ɫ��ƽ��.bmp");
}

void ImgProcess::colorBalance(int deltaR, int deltaG, int deltaB)
{
    Mat colorBalanceImg; // ���ڴ���ɫ��ƽ����ͼ��
    colorBalanceImg.create(M.size(), M.type());
    colorBalanceImg = cv::Scalar::all(0); // ��ʼ����ȫ����0
    int delta[3] = { deltaB,deltaG,deltaR }; // opencv��RGB˳���Ƿ���

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
    SaveImage(colorBalanceImg, "�Զ���ɫ��ƽ��ͼ��.bmp");
}

///////////////////////////////////////////////////////

/*
// ���溯��Ϊͼ��ƽ�ơ����ź���ת����������ͼ�������غ��������� TODO

void ImgProcess::imgMove(int dx, int dy)
{
    // �������������ֱ��ͼ�����ƽ�ơ����ź���ת������任������
    //���ҡ��·��ֱ�ƽ�ƣ�dx��dy������
    Mat movedImg;
    Vec3b* p; // �൱��uchar���͵�,����Ϊ3��vector����

    // �ı���ͼ���С
    int rows = Rows + abs(dy); //���ͼ��Ĵ�С
    int cols = Cols + abs(dx);
    movedImg.create(rows, cols, M.type());
    for (int i = 0; i < rows; i++)
    {
        p = movedImg.ptr<Vec3b>(i); // ��ȡͼ����ָ��
        for (int j = 0; j < cols; j++)
        {
            // λ������ӳ��������->ԭ����+λ����
            int x = j - dx;
            int y = i - dy;
            // ��֤ӳ����������ԭͼ��Χ��
            if (x >= 0 && y >= 0 && x < cols && y < rows)
                p[j] = M.ptr<Vec3b>(y)[x];
        }
    }
    SaveImage(movedImg, "ͼ��ƽ�Ʊ任.bmp");
}
// TODO
void ImgProcess::imgZoom(double xRate, double yRate)
{
    // ����ͼ���ľ����С
    int rows = Rows * xRate;
    int cols = Cols * yRate;
    Mat zoomedImg = Mat::zeros(rows, cols, M.type());
    uchar* pDst = zoomedImg.data;

    // �������ű任����
    Mat T = (Mat_<double>(3, 3) << xRate, 0, 0, 0, yRate, 0, 0, 0, 1);
    Mat invT = T.inv(); // ��������

    // ע��ָ��������ʽ�����˱任��λ�������磺{��, 0, 0; 0, ��, 0; 0, 0, 1}
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1); // ��i��j�е���������(j,i)
            Mat src_uv = dst_xy * invT;
            double u = src_uv.at<double>(0, 0); // ԭͼ��ĺ����꣬��Ӧͼ�������
            double v = src_uv.at<double>(0, 1);

            // ˫���Բ�ֵ����ֵ���㲻���ڵ����ص�
            if (u >= 0 && v >= 0 && u <= Cols - 1 && v <= Rows - 1) {
                // ��ӳ�䵽���������ص�����꣬�ֱ���������ȡ��
                int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u);
                // ��������ƫ���dv��du�ֱ�Ϊ�С���С������
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
    SaveImage(zoomedImg, "ͼ�����ű任.bmp");
}

void ImgProcess::imgRotate(double angle)
{
    // ��ת�����У�angleΪ˳ʱ����ת�ĽǶ�
    angle = angle * CV_PI / 180;
    int rows = round(fabs(Rows * cos(angle)) + fabs(Cols * sin(angle)));
    int cols = round(fabs(Cols * cos(angle)) + fabs(Rows * sin(angle)));
    // �������
    Mat rotImg = Mat::zeros(rows, cols, M.type());
    uchar* pDst = rotImg.data;

    //������ת�任���󣬻�������ͬ��ת
    Mat T1 = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.5 * Cols, 0.5 * Rows, 1.0);
    Mat T2 = (Mat_<double>(3, 3) << cos(angle), -sin(angle), 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0, 1.0);
    // �ѿ�������ӳ��
    double t3[3][3] = { { 1.0, 0.0, 0.0 },{ 0.0, -1.0, 0.0 },{ 0.5 * rotImg.cols, 0.5 * rotImg.rows ,1.0 } };
    Mat T3 = Mat(3.0, 3.0, CV_64FC1, t3);
    Mat T = T1 * T2 * T3;
    Mat invT = T.inv();//����

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1); // ��i��j�е���������(j,i)
            Mat src_uv = dst_xy * invT;
            double u = src_uv.at<double>(0, 0); // ԭͼ��ĺ����꣬��Ӧͼ�������
            double v = src_uv.at<double>(0, 1);

            //˫���Բ�ֵ��
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
    SaveImage(rotImg, "ͼ����ת�任.bmp");
}

// ����ĺ���Ϊ����ͼ�������غ���������

void ImgProcess::imgAutoCorr()
{
    imwrite("pre_pic.bmp", M); // ת��Ϊ�Ҷ�ͼ��
    Mat binImg = imread("pre_pic.bmp", IMREAD_GRAYSCALE);
    int deno = 0, nume = 0; // ��������غ�����ʽ����ĸ�̶�Ϊdeno������nume��һֱ�仯
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            // ����ѭ��Ƕ�ף�����ԭ����ͼ�񣬲���������ֵƽ����
            deno = deno + (int)binImg.at<uchar>(i, j) * (int)binImg.at<uchar>(i, j);
        }
    }
    // ����Ŀ��ͼ��
    Mat autoCorrImg;
    autoCorrImg.create(M.size(), binImg.type());
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            // ѭ�������µ�ͼ��
            for (int m = 0; m < Rows; m++) {
                for (int n = 0; n < Cols; n++) {
                    // ����ѭ��Ƕ�׵õ����ӣ��˴������m+i��n+j�൱�����ϵ�i+x��j+y
                    if (m + i > Rows - 1 || n + j > Cols - 1) {
                        nume = 0;
                    }
                    else {
                        nume = nume + (int)binImg.at<uchar>(m, n) * (int)binImg.at<uchar>(m + i, n + j); // �����ۼ�
                    }
                }
            }
            autoCorrImg.at<uchar>(i, j) = saturate_cast<uchar>(nume / deno); // ���Ӽ�����ϣ��Ե�ǰ���ظ�ֵ
            nume = 0; // ���ӹ�0���ȴ��´μ���
        }
    }
    SaveImage(autoCorrImg, "����غ���ͼ��.bmp");
}
*/