#include "imageshop.h"
#include "ui_imageshop.h"

ImageShop::ImageShop(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ImageShop)
{
    ui->setupUi(this);
}

ImageShop::~ImageShop()
{
    delete ui;
}

/*HomeworkOne*/

void ImageShop::showImage(cv::Mat img)
{
    QImage qImage((const uchar *)img.data,img.cols,img.rows,img.step,QImage::Format_RGB888);
    QSize size;
    size.setHeight(qImage.height());
    size.setWidth(qImage.width());
    ui->imageLabel->resize(size);
    ui->imageLabel->setPixmap(QPixmap::fromImage(qImage));
}

void ImageShop::rotate(double angle,cv::Mat src)
{
    cv::Mat dst = src.clone();
    int len = std::max(src.cols, src.rows);
    cv::Size src_sz = src.size();
    cv::Point2f center(len / 2., len / 2.);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(src, dst, rot_mat, src_sz,1,0,cv::Scalar(255,255,255));
    showImage(dst);
}

void ImageShop::flip(int flipCode,cv::Mat src)
{
    cv::Mat img(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    if(flipCode == 0){
        for (int x = 0; x < src.cols; x++)
        {
            for (int y = 0; y < src.rows; y++)
            {
                img.at<cv::Vec3b>(cv::Point(x, src.rows-1-y))[0] = src.at<cv::Vec3b>(cv::Point(x, y))[0];
                img.at<cv::Vec3b>(cv::Point(x, src.rows-1-y))[1] = src.at<cv::Vec3b>(cv::Point(x, y))[1];
                img.at<cv::Vec3b>(cv::Point(x, src.rows-1-y))[2] = src.at<cv::Vec3b>(cv::Point(x, y))[2];
            }
        }
    }
    else if(flipCode == 1){
        for (int x = 0; x < src.cols; x++)
        {
            for (int y = 0; y < src.rows; y++)
            {
                img.at<cv::Vec3b>(cv::Point(src.cols-1-x, y))[0] = src.at<cv::Vec3b>(cv::Point(x, y))[0];
                img.at<cv::Vec3b>(cv::Point(src.cols-1-x, y))[1] = src.at<cv::Vec3b>(cv::Point(x, y))[1];
                img.at<cv::Vec3b>(cv::Point(src.cols-1-x, y))[2] = src.at<cv::Vec3b>(cv::Point(x, y))[2];
            }
        }
    }
    this->image = img;
    showImage(img);
}

void ImageShop::cut(int cutCode,cv::Point start,cv::Point stop,cv::Mat src)
{
    cv::Mat img(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    if(cutCode == 1){
        for (int x = 0; x < src.cols; x++)
        {
            for (int y = 0; y < src.rows; y++)
            {
                if ((x >= start.x && y >= start.y) && (x <= stop.x && y <= stop.y))
                {
                    img.at<cv::Vec3b>(cv::Point(x, y))[0] = src.at<cv::Vec3b>(cv::Point(x, y))[0];
                    img.at<cv::Vec3b>(cv::Point(x, y))[1] = src.at<cv::Vec3b>(cv::Point(x, y))[1];
                    img.at<cv::Vec3b>(cv::Point(x, y))[2] = src.at<cv::Vec3b>(cv::Point(x, y))[2];
                }
            }
        }
    }
    else if(cutCode == 2){
        int radius = (int)sqrt((start.x - stop.x)*(start.x - stop.x) + (start.y - stop.y)*(start.y - stop.y))/2;
        cv::Point center((start.x + stop.x) / 2,(start.y + stop.y) / 2);
        for (int x = 0; x < src.cols; x++)
        {
            for (int y = 0; y < src.rows; y++)
            {
                int temp = ((x - center.x) * (x - center.x) + (y - center.y) *(y - center.y));
                if (temp < (radius * radius))
                {
                    img.at<cv::Vec3b>(cv::Point(x, y))[0] = src.at<cv::Vec3b>(cv::Point(x, y))[0];
                    img.at<cv::Vec3b>(cv::Point(x, y))[1] = src.at<cv::Vec3b>(cv::Point(x, y))[1];
                    img.at<cv::Vec3b>(cv::Point(x, y))[2] = src.at<cv::Vec3b>(cv::Point(x, y))[2];
                }
            }
        }
    }
    this->image = img;
    showImage(img);
}

/*HomeworkTwo*/

void ImageShop::RGB2HLS(double *h_hls, double *l_hls, double *s_hls, int r_rgb, int g_rgb, int b_rgb)
{
    double dr = (double)r_rgb/255;
    double dg = (double)g_rgb/255;
    double db = (double)b_rgb/255;
    double cmax = MAX(dr, MAX(dg, db));
    double cmin = MIN(dr, MIN(dg, db));
    double cdes = cmax - cmin;
    double hh, ll, ss;

    ll = (cmax+cmin)/2;
    if(cdes){
        if(ll <= 0.5)
            ss = (cmax-cmin)/(cmax+cmin);
        else
            ss = (cmax-cmin)/(2-cmax-cmin);

        if(cmax == dr)
            hh = (0+(dg-db)/cdes)*60;
        else if(cmax == dg)
            hh = (2+(db-dr)/cdes)*60;
        else// if(cmax == b)
            hh = (4+(dr-dg)/cdes)*60;
        if(hh<0)
            hh+=360;
    }else
        hh = ss = 0;

    *h_hls = hh;
    *l_hls = ll;
    *s_hls = ss;
}

void ImageShop::HLS2RGB(int *r_rgb, int *g_rgb, int *b_rgb, double h_hls, double l_hls, double s_hls)
{
    double cmax,cmin;

    if(l_hls <= 0) l_hls = 0;
    else if(l_hls >= 1) l_hls = 1;
    if(s_hls <= 0) s_hls = 0;
    else if(s_hls >= 1) s_hls = 1;

    if(l_hls <= 0.5)
        cmax = l_hls*(1+s_hls);
    else
        cmax = l_hls*(1-s_hls)+s_hls;
    cmin = 2*l_hls-cmax;

    if(s_hls == 0){
        *r_rgb = *g_rgb = *b_rgb = l_hls*255;
    }else{
        *r_rgb = cv::saturate_cast<uchar>(HLS2RGBvalue(cmin,cmax,h_hls+120)*255);
        *g_rgb = cv::saturate_cast<uchar>(HLS2RGBvalue(cmin,cmax,h_hls)*255);
        *b_rgb = cv::saturate_cast<uchar>(HLS2RGBvalue(cmin,cmax,h_hls-120)*255);
    }
}

double ImageShop::HLS2RGBvalue(double h1,double h2, double hue)
{
    if(hue > 360)
        hue -= 360;
    else if(hue < 0)
        hue += 360;
    if(hue < 60)
        return h1+(h2-h1)*hue/60;
    else if(hue < 180)
        return h2;
    else if(hue < 240)
        return h1+(h2-h1)*(240-hue)/60;
    else
        return h1;
}

void ImageShop::colorChange(double hue,double lightness,double saturation,cv::Mat src){
    cv::Mat img(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    for (int x = 0; x < src.cols; x++)
    {
        for (int y = 0; y < src.rows; y++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = src.at<cv::Vec3b>(cv::Point(x, y))[0];
            int g_rgb = src.at<cv::Vec3b>(cv::Point(x, y))[1];
            int b_rgb = src.at<cv::Vec3b>(cv::Point(x, y))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls+hue, *l_hls+lightness, *s_hls+saturation);
            img.at<cv::Vec3b>(cv::Point(x, y))[0] = r_rgb;
            img.at<cv::Vec3b>(cv::Point(x, y))[1] = g_rgb;
            img.at<cv::Vec3b>(cv::Point(x, y))[2] = b_rgb;
        }
    }
    showImage(img);
}

void ImageShop::colorHalftone(cv::Mat src){
    cv::Mat img(src.rows, src.cols, src.type(), cv::Scalar(255, 255, 255));
    cv::Mat img1(src.rows, src.cols, src.type(), cv::Scalar(255, 255, 255));
    cv::Mat img2(src.rows, src.cols, src.type(), cv::Scalar(255, 255, 255));
    cv::Mat img3(src.rows, src.cols, src.type(), cv::Scalar(255, 255, 255));
    for (int x = 5; x < img1.cols-5; x=x+10)
    {
        for (int y = 5; y < img1.rows-5; y=y+10)
        {
            int r_rgb = src.at<cv::Vec3b>(cv::Point(x, y))[0];
            double l_hls = (double)r_rgb/255;

            for (int i = x-5; i < x+5; i++)
            {
                for (int j = y-5; j < y+5; j++)
                {
                    double temp = ((x - i) * (x - i) + (y - j) *(y - j));
                    if (temp < ((1-l_hls)*25))
                    {
                        img1.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
                    }
                }
            }
        }
    }
    rotate(-1,img1);
    for (int x = 5; x < img2.cols-5; x=x+10)
    {
        for (int y = 5; y < img2.rows-5; y=y+10)
        {
            int g_rgb = src.at<cv::Vec3b>(cv::Point(x, y))[1];
            double l_hls = (double)g_rgb/255;
            for (int i = x-5; i < x+5; i++)
            {
                for (int j = y-5; j < y+5; j++)
                {
                    double temp = ((x - i) * (x - i) + (y - j) *(y - j));
                    if (temp < ((1-l_hls)*25))
                    {
                        img2.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
                    }
                }
            }
        }
    }
    rotate(1,img2);
    for (int x = 5; x < img3.cols-5; x=x+10)
    {
        for (int y = 5; y < img3.rows-5; y=y+10)
        {
            int b_rgb = src.at<cv::Vec3b>(cv::Point(x, y))[2];
            double l_hls = (double)b_rgb/255;
            for (int i = x-5; i < x+5; i++)
            {
                for (int j = y-5; j < y+5; j++)
                {
                    double temp = ((x - i) * (x - i) + (y - j) *(y - j));
                    if (temp < ((1-l_hls)*25))
                    {
                        img3.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;
                    }
                }
            }
        }
    }
    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            img.at<cv::Vec3b>(cv::Point(x, y))[0] = img1.at<cv::Vec3b>(cv::Point(x, y))[0];
            img.at<cv::Vec3b>(cv::Point(x, y))[1] = img2.at<cv::Vec3b>(cv::Point(x, y))[1];
            img.at<cv::Vec3b>(cv::Point(x, y))[2] = img3.at<cv::Vec3b>(cv::Point(x, y))[2];
        }
    }
    showImage(img);
}

/*HomeworkThree*/

void ImageShop::segmentalLinearTransform(int a,int b,int c,int d,cv::Mat src){
    cv::Mat dst = src.clone();
    if((a<b)&&(c<d)&&(a>0)&&(b<255)){
        for (int i = 0; i < dst.rows; i++)
        {
            for (int j = 0; j < dst.cols; j++)
            {
                double *h_hls = new double();
                double *s_hls = new double();
                double *l_hls = new double();
                int r_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[0];
                int g_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[1];
                int b_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[2];
                RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
                double value = cv::saturate_cast<uchar>((*l_hls)*255);
                if(value < c){
                    value= cv::saturate_cast<uchar>( (c/a) * value );
                }
                else if(c <= value && value <= d ){
                    value = (cv::saturate_cast<uchar>((d - c) / (b - a) * (value - a) + c));
                }
                else{
                    value = (cv::saturate_cast<uchar>((255-d) / (255-b) * (value - b) + d));
                }
                *l_hls = (double)value/255;
                HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
                dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
                dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
                dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;
            }
        }
    }
    showImage(dst);
}

void ImageShop::logarithmicTransform(double a,double b,double c,cv::Mat src){
    cv::Mat dst = src.clone();
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            double value = cv::saturate_cast<uchar>((*l_hls)*255);
            value = a+log((double)(1 + value))/(b*log(c));
            *l_hls = value;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;
        }
    }
    showImage(dst);
}

void ImageShop::exponentialTransform(double a,double b,double c,cv::Mat src){
    cv::Mat dst = src.clone();
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            double value = cv::saturate_cast<uchar>((*l_hls)*255);
            value = pow(b,c*(value-a))-1;
            *l_hls = value;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;
        }
    }
    showImage(dst);
}

void ImageShop::gammaCorrectTransform(double r,cv::Mat src){
    cv::Mat dst = src.clone();
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            double value = cv::saturate_cast<uchar>((*l_hls)*255);
            value=pow(value/255.0,1/r);
            *l_hls = value;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;
        }
    }
    showImage(dst);
}
void ImageShop::getHistogram(cv::Mat src){
    cv::cvtColor(src,src,cv::COLOR_RGB2GRAY);
    cv::Mat hist;
    int imgNum = 1;
    int histDim = 1;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::calcHist(&src, imgNum, 0, cv::Mat(), hist, histDim, &histSize, &histRange, uniform, accumulate);
    int scale = 2;
    cv::Mat histImg(cv::Size(histSize*scale, histSize), CV_8UC1);
    uchar* pImg = nullptr;
    for (size_t i = 0; i < histImg.rows; i++)
    {
        pImg = histImg.ptr<uchar>(i);
        for (size_t j = 0; j < histImg.cols; j++)
        {
            pImg[j] = 0;
        }
    }
    double maxValue = 0;
    cv::minMaxLoc(hist, 0, &maxValue, 0, 0);
    int histHeight = 256;
    float* p = hist.ptr<float>(0);
    for (size_t i = 0; i < histSize; i++)
    {
        float bin_val = p[i];
        int intensity = cvRound(bin_val*histHeight / maxValue);
        for (size_t j = 0; j < scale; j++)
        {
            cv::line(histImg, cv::Point(i*scale + j , histHeight - intensity), cv::Point(i*scale + j, histHeight - 1), 255);
        }

    }
    QImage qImage((const uchar *)histImg.data,histImg.cols,histImg.rows,histImg.step,QImage::Format_Grayscale8);
    QSize size;
    size.setHeight(qImage.height());
    size.setWidth(qImage.width());
    ui->label->resize(size);
    ui->label->setPixmap(QPixmap::fromImage(qImage));
 }

std::vector<double> ImageShop::getAccumulateHistogram(cv::Mat src){
    std::vector<double> dst_hist(256,0);
    std::vector<double> dst_s(256,0);
    cv::Mat dst = src.clone();
    int total=dst.rows*dst.cols;
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            double value = cv::saturate_cast<uchar>((*l_hls)*255);
            dst_hist[value]++;
        }
    }
    float now=0.0;
    for(int i=0;i<dst_s.size();i++){
        now+=dst_hist[i]/total;
        dst_s[i]=now;
    }
    return dst_s;
}

void ImageShop::Equalization(cv::Mat src){
    std::vector<double> dst_s=this->getAccumulateHistogram(src);
    cv::Mat dst = src.clone();
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            double value = cv::saturate_cast<uchar>((*l_hls)*255);
            value=cvRound(255*dst_s[value]);
            *l_hls = (double)value/255;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;
        }
    }
    getHistogram(dst);
    showImage(dst);
}

void ImageShop::gmlSpecification(std::vector<double> learn_s){
    std:: vector<double> dst_s=this->getAccumulateHistogram(this->image);
    cv::Mat lut(1, 256, CV_8U);
    int mj=1,flag=0;
    for(int i=0;i<256;i++){
        if(learn_s[i]>0){
            for(;mj<256;mj++){
            if(abs(learn_s[i]-dst_s[mj])>abs(learn_s[i]-dst_s[mj-1])){
                for(int m=flag;m<mj;m++){
                    lut.at<uchar>(m) = static_cast<uchar>(i);
                }
                flag=mj;
                break;
            }
            }
        }

    }
    cv::Mat dst = this->image.clone();
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            int value = cv::saturate_cast<uchar>((*l_hls)*255);
            value=lut.at<uchar>(value);
            *l_hls = (double)value/255;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;

        }
    }
    getHistogram(dst);
    showImage(dst);
}

void ImageShop::smlSpecification(std::vector<double> learn_s){
    std:: vector<double> dst_s=this->getAccumulateHistogram(this->image);
    cv::Mat lut(1, 256, CV_8U);
    for(int i=0;i<256;i++){
        int tmp=INT_MAX;
        int index;
        for(int j=0;j<256;j++){
            if(learn_s[j]>0){
                if(abs(dst_s[i]-learn_s[j])<tmp){
                    tmp=abs(dst_s[i]-learn_s[j]);
                    index=j;
                }
            }
        }
        lut.at<uchar>(i) = static_cast<uchar>(index);
    }
    cv::Mat dst = this->image.clone();
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = this->image.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            int value = cv::saturate_cast<uchar>((*l_hls)*255);
            value=lut.at<uchar>(value);
            *l_hls = (double)value/255;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;

        }
    }
    getHistogram(dst);
    showImage(dst);
}

/*HomeworkFour*/

double **ImageShop::getGuassionArray(int size) {
    double sum = 0.0;
    //double sigma = 0.3*((size-1)*0.5 - 1) + 0.8;
    int center = (size+1)/2;
    double **arr = new double*[size];
    for (int i = 0; i < size; i++)
        arr[i] = new double[size];
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; ++j) {
            arr[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2*2* 2));
            sum += arr[i][j];
        }
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            arr[i][j] /= sum;
    return arr;
}

void ImageShop::medianFilter (cv::Mat src,int size){
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    for (int i = 0; i < src.rows; ++i){
        for (int j = 0; j < src.cols; ++j) {
            int bound = (size-1)/2;
            if ((i - bound) > 0 && (i + bound) < src.rows && (j - bound) > 0 && (j + bound) < src.cols) {
                std::vector<int> r_arr;
                std::vector<int> g_arr;
                std::vector<int> b_arr;
                for (int x = 0; x < size; ++x) {
                    for (int y = 0; y < size; ++y) {
                           r_arr.push_back((src.at<cv::Vec3b>(i + bound - x, j + bound - y)[0]));
                           g_arr.push_back((src.at<cv::Vec3b>(i + bound - x, j + bound - y)[1]));
                           b_arr.push_back((src.at<cv::Vec3b>(i + bound - x, j + bound - y)[2]));
                    }
                }
                std::sort(r_arr.begin(),r_arr.end());
                std::sort(g_arr.begin(),g_arr.end());
                std::sort(b_arr.begin(),b_arr.end());
                dst.at<cv::Vec3b>(i, j)[0] = r_arr.at(r_arr.size()/2);
                dst.at<cv::Vec3b>(i, j)[1] = g_arr.at(r_arr.size()/2);
                dst.at<cv::Vec3b>(i, j)[2] = b_arr.at(r_arr.size()/2);
            }
            else{
                dst.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(i , j)[0];
                dst.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(i , j)[1];
                dst.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(i , j)[2];
            }
        }
    }
    getHistogram(dst);
    showImage(dst);
}
void ImageShop::GaussianFilter(cv::Mat src,int size){
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    double **arr = getGuassionArray(size);
    for (int i = 0; i < src.rows; ++i){
        for (int j = 0; j < src.cols; ++j) {
            int bound = (size-1)/2;
            if ((i - bound) > 0 && (i + bound) < src.rows && (j - bound) > 0 && (j + bound) < src.cols) {
                for (int x = 0; x < size; ++x) {
                    for (int y = 0; y < size; ++y) {
                            dst.at<cv::Vec3b>(i, j)[0] += arr[x][y] * src.at<cv::Vec3b>(i + bound - x, j + bound - y)[0];
                            dst.at<cv::Vec3b>(i, j)[1] += arr[x][y] * src.at<cv::Vec3b>(i + bound - x, j + bound - y)[1];
                            dst.at<cv::Vec3b>(i, j)[2] += arr[x][y] * src.at<cv::Vec3b>(i + bound - x, j + bound - y)[2];
                    }
                }
            }
            else{
                dst.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(i , j)[0];
                dst.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(i , j)[1];
                dst.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(i , j)[2];
            }
        }
    }
    getHistogram(dst);
    showImage(dst);
}
void ImageShop::sharpenFilter(cv::Mat src){
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    for (int i = 0; i < src.rows; ++i){
        for (int j = 0; j < src.cols; ++j) {
            if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
                dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(
                                            5*src.at<cv::Vec3b>(i , j)[0]
                                            -src.at<cv::Vec3b>(i + 1, j)[0]
                                            -src.at<cv::Vec3b>(i - 1, j)[0]
                                            -src.at<cv::Vec3b>(i , j + 1)[0]
                                            -src.at<cv::Vec3b>(i , j - 1)[0]);
                dst.at<cv::Vec3b>(i, j)[1] =cv::saturate_cast<uchar>(
                                            5*src.at<cv::Vec3b>(i , j)[1]
                                            -src.at<cv::Vec3b>(i + 1, j)[1]
                                            -src.at<cv::Vec3b>(i - 1, j)[1]
                                            -src.at<cv::Vec3b>(i , j + 1)[1]
                                            -src.at<cv::Vec3b>(i , j - 1)[1]);
                dst.at<cv::Vec3b>(i, j)[2] =cv::saturate_cast<uchar>(
                                            5*src.at<cv::Vec3b>(i , j)[2]
                                            -src.at<cv::Vec3b>(i + 1, j)[2]
                                            -src.at<cv::Vec3b>(i - 1, j)[2]
                                            -src.at<cv::Vec3b>(i , j + 1)[2]
                                            -src.at<cv::Vec3b>(i , j - 1)[2]);
            }
            else{
                dst.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(i , j)[0];
                dst.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(i , j)[1];
                dst.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(i , j)[2];
            }
        }
    }
    getHistogram(dst);
    showImage(dst);
}

void ImageShop::raditionBlur(cv::Mat src){
    cv::Mat dst = src.clone();
    float R;
    float angle;
    cv::Point Center(src.cols/2, src.rows/2);
    float t1, t2, t3;
    int new_x, new_y;
    int Num=20;
    for (int y=0; y<src.rows; y++)
    {
        for (int x=0; x<src.cols; x++)
        {
            t1=0; t2=0; t3=0;
            R=sqrt((y-Center.y)*(y-Center.y)+(x-Center.x)*(x-Center.x));
            angle=atan2((float)(y-Center.y), (float)(x-Center.x));
            for (int mm=0; mm<Num; mm++)
            {
                float tmR=R-mm>0 ? R-mm : 0.0;
                new_x=tmR*cos(angle)+Center.x;
                new_y=tmR*sin(angle)+Center.y;

                if(new_x<0)       new_x=0;
                if(new_x>src.cols-1) new_x=src.cols-1;
                if(new_y<0)       new_y=0;
                if(new_y>src.rows-1) new_y=src.rows-1;

                t1=t1+src.at<cv::Vec3b>(new_y, new_x)[0];
                t2=t2+src.at<cv::Vec3b>(new_y, new_x)[1];
                t3=t3+src.at<cv::Vec3b>(new_y, new_x)[2];

            }

            dst.at<cv::Vec3b>(y, x)[0]=t1/Num;
            dst.at<cv::Vec3b>(y, x)[1]=t2/Num;
            dst.at<cv::Vec3b>(y, x)[2]=t3/Num;

        }
    }
    getHistogram(dst);
    showImage(dst);
}

void ImageShop::snowBlur(cv::Mat src){
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    cv::RNG rng(cv::getCPUTickCount());
    for (int i=0; i<1000; i++)
    {
        int D = rng.uniform(0, src.cols);
        int x = round(D);
        D = rng.uniform(0, src.rows);
        int y = round(D);
        int size = rng.uniform(1, 5);
        int s = round(size);
        for (int l = 0; l < s; ++l) {
            for (int k = 0; k < s; ++k) {
                int new_x=x+l;
                int new_y=y+k;

                if(new_x<0)       new_x=0;
                if(new_x>src.rows) new_x=src.rows;
                if(new_y<0)       new_y=0;
                if(new_y>src.cols) new_y=src.cols;
                dst.at<cv::Vec3b>(new_x, new_y)[0]=255;
                dst.at<cv::Vec3b>(new_x, new_y)[1]=255;
                dst.at<cv::Vec3b>(new_x, new_y)[2]=255;
            }
        }

    }
    for (int x=0; x<dst.rows; x++)
    {
        for (int y=0; y<dst.cols; y++)
        {
            if ((x - 1) > 0 && (x + 1) < dst.rows && (y - 1) > 0 && (y + 1) < dst.cols) {
                dst.at<cv::Vec3b>(x, y)[0]=cv::saturate_cast<uchar>(
                            (dst.at<cv::Vec3b>(x , y)[0]
                            +dst.at<cv::Vec3b>(x + 1, y+1)[0]
                            +dst.at<cv::Vec3b>(x - 1, y-1)[0])/3);
                dst.at<cv::Vec3b>(x, y)[1]=cv::saturate_cast<uchar>(
                            (dst.at<cv::Vec3b>(x , y)[1]
                            +dst.at<cv::Vec3b>(x + 1, y+1)[1]
                            +dst.at<cv::Vec3b>(x - 1, y-1)[1])/3);
                dst.at<cv::Vec3b>(x, y)[2]=cv::saturate_cast<uchar>(
                            (dst.at<cv::Vec3b>(x , y)[2]
                            +dst.at<cv::Vec3b>(x + 1, y+1)[2]
                            +dst.at<cv::Vec3b>(x - 1, y-1)[2])/3);
            }

        }
    }
    for (int x=0; x<src.rows; x++)
    {
        for (int y=0; y<src.cols; y++)
        {
            if ((x - 1) > 0 && (x + 1) < src.rows && (y - 1) > 0 && (y + 1) < src.cols) {
                dst.at<cv::Vec3b>(x, y)[0]=cv::saturate_cast<uchar>(
                            dst.at<cv::Vec3b>(x , y)[0]
                            +src.at<cv::Vec3b>(x , y)[0]
                            -(dst.at<cv::Vec3b>(x , y)[0]*src.at<cv::Vec3b>(x , y)[0])/255);
                dst.at<cv::Vec3b>(x, y)[1]=cv::saturate_cast<uchar>(
                            dst.at<cv::Vec3b>(x , y)[1]
                            +src.at<cv::Vec3b>(x , y)[1]
                            -(dst.at<cv::Vec3b>(x , y)[1]*src.at<cv::Vec3b>(x , y)[1])/255);
                dst.at<cv::Vec3b>(x, y)[2]=cv::saturate_cast<uchar>(
                            dst.at<cv::Vec3b>(x , y)[2]
                            +src.at<cv::Vec3b>(x , y)[2]
                            -(dst.at<cv::Vec3b>(x , y)[2]*src.at<cv::Vec3b>(x , y)[2])/255);
            }

        }
    }
    getHistogram(dst);
    showImage(dst);
}
/*HomeworkFive*/
void ImageShop::BLPF(cv::Mat src,int N, int D0){
    cv::Mat img;
    cv::cvtColor(src, img, CV_RGB2GRAY);
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::Mat lowTemp = cv::Mat(complexI.size(),complexI.type());
    //ButterWorth低通滤波器
    int cx = complexI.cols / 2;
    int cy = complexI.rows / 2;
    int state=-1;
    double tempD;
    double h;
    //D0 = 0.3 * cv::min(lowTemp.rows, lowTemp.cols) / 2.0;
    for (int i = 0; i < lowTemp.rows; i++)
    {
        float* ptr = lowTemp.ptr<float>(i);
        for (int j = 0; j < lowTemp.cols; j++)
        {
            if (i > cy && j > cx)
            {
                state = 3;
            }
            else if (i > cy)
            {
                state = 1;
            }
            else if (j > cx)
            {
                state = 2;
            }
            else
            {
                state = 0;
            }

            switch (state)
            {
            case 0:
                tempD = (double)(i * i + j * j); tempD = sqrt(tempD); break;
            case 1:
                tempD = (double)((lowTemp.rows - i) * (lowTemp.rows - i) + j * j); tempD = sqrt(tempD); break;
            case 2:
                tempD = (double)(i * i + (lowTemp.cols - j) * (lowTemp.cols - j)); tempD = sqrt(tempD); break;
            case 3:
                tempD = (double)((lowTemp.rows - i) * (lowTemp.rows - i) + (lowTemp.cols - j) * (lowTemp.cols - j)); tempD = sqrt(tempD); break;
            default:
                break;
            }
            //tempD = exp(-0.5 * pow(tempD / D0, 2));
            h =  1/ (1 + pow(tempD / D0, 2 * N));
            ptr[j*2]= h;
            ptr[2*j+1]= h;
        }
    }
    cv::mulSpectrums(complexI, lowTemp, complexI,0);
    cv::idft(complexI, complexI, cv::DFT_SCALE);
    cv::split(complexI, planes);
    cv::Mat dft_filter_img;
    dft_filter_img.create(complexI.size(), CV_8UC1);
    planes[0].convertTo(dft_filter_img, CV_8UC1);
    dft_filter_img = dft_filter_img(cv::Rect(0, 0, img.cols, img.rows));
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            int value = dft_filter_img.at<uchar>(j, i);
            *l_hls = (double)value/255;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;

        }
    }
    getHistogram(dst);
    showImage(dst);
}

void ImageShop::BHPF(cv::Mat src,int N, int D0){
    cv::Mat img;
    cv::cvtColor(src, img, CV_RGB2GRAY);
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::Mat lowTemp = cv::Mat(complexI.size(),complexI.type());
    //创建低通滤波器
    int cx = complexI.cols / 2;
    int cy = complexI.rows / 2;
    int state=-1;
    double tempD;
    double h;
    for (int i = 0; i < lowTemp.rows; i++)
    {
        float* ptr = lowTemp.ptr<float>(i);
        for (int j = 0; j < lowTemp.cols; j++)
        {
            if (i > cy && j > cx)
            {
                state = 3;
            }
            else if (i > cy)
            {
                state = 1;
            }
            else if (j > cx)
            {
                state = 2;
            }
            else
            {
                state = 0;
            }

            switch (state)
            {
            case 0:
                tempD = (double)(i * i + j * j); tempD = sqrt(tempD); break;
            case 1:
                tempD = (double)((lowTemp.rows - i) * (lowTemp.rows - i) + j * j); tempD = sqrt(tempD); break;
            case 2:
                tempD = (double)(i * i + (lowTemp.cols - j) * (lowTemp.cols - j)); tempD = sqrt(tempD); break;
            case 3:
                tempD = (double)((lowTemp.rows - i) * (lowTemp.rows - i) + (lowTemp.cols - j) * (lowTemp.cols - j)); tempD = sqrt(tempD); break;
            default:
                break;
            }
            h =  1/ (1 + pow(D0 / tempD, 2 * N))+1;
            ptr[j*2]= h;
            ptr[2*j+1]= h;
        }
    }
    cv::mulSpectrums(complexI, lowTemp, complexI,0);
    cv::idft(complexI, complexI, cv::DFT_SCALE);
    cv::split(complexI, planes);
    cv::Mat dft_filter_img;
    dft_filter_img.create(complexI.size(), CV_8UC1);
    planes[0].convertTo(dft_filter_img, CV_8UC1);
    dft_filter_img = dft_filter_img(cv::Rect(0, 0, img.cols, img.rows));
    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            double *h_hls = new double();
            double *s_hls = new double();
            double *l_hls = new double();
            int r_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[0];
            int g_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[1];
            int b_rgb = src.at<cv::Vec3b>(cv::Point(i, j))[2];
            RGB2HLS(h_hls, l_hls, s_hls, r_rgb, g_rgb, b_rgb);
            int value = dft_filter_img.at<uchar>(j, i);
            *l_hls = (double)value/255;
            HLS2RGB(&r_rgb, &g_rgb, &b_rgb, *h_hls, *l_hls, *s_hls);
            dst.at<cv::Vec3b>(cv::Point(i, j))[0] = r_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[1] = g_rgb;
            dst.at<cv::Vec3b>(cv::Point(i, j))[2] = b_rgb;

        }
    }
    Equalization(dst);
}

/*SLOT*/

void ImageShop::on_actionOpen_O_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open file","","Image Files(*.jpg *.png *.bmp *.jpeg)");
    if(!fileName.isEmpty()){
        this->fileName = fileName;
        QTextCodec *code = QTextCodec::codecForName("gb18030");
        std::string name = code->fromUnicode(fileName).data();
        this->image = cv::imread(name,CV_LOAD_IMAGE_UNCHANGED);
        cv::cvtColor(this->image,this->image,CV_BGR2RGB);
        if(this->image.data){
            ui->spinBox_rotate->setValue(0);
            ui->horizontalSlider_rotate->setValue(0);

            ui->spinBox_saturation->setValue(0);
            ui->horizontalSlider_saturation->setValue(0);
            ui->spinBox_hue->setValue(0);
            ui->horizontalSlider_hue->setValue(0);
            ui->spinBox_lightness->setValue(0);
            ui->horizontalSlider_lightness->setValue(0);

            ui->spinBox_transform_a->setValue(0);
            ui->horizontalSlider_transform_a->setValue(0);
            ui->spinBox_transform_b->setValue(0);
            ui->horizontalSlider_transform_b->setValue(0);
            ui->spinBox_transform_c->setValue(0);
            ui->horizontalSlider_transform_c->setValue(0);
            ui->spinBox_transform_d->setValue(0);
            ui->horizontalSlider_transform_d->setValue(0);
            ui->doubleSpinBox_logarithmic_transform_a->setValue(0);
            ui->doubleSpinBox_logarithmic_transform_b->setValue(0);
            ui->doubleSpinBox_logarithmic_transform_c->setValue(0);
            ui->doubleSpinBox_exponential_trasform_a->setValue(0);
            ui->doubleSpinBox_exponential_trasform_b->setValue(0);
            ui->doubleSpinBox_exponential_trasform_c->setValue(0);
            ui->doubleSpinBox_gamma_correct->setValue(0);

            showImage(this->image);
            getHistogram(this->image);
        }
        else{
            QMessageBox::information(this, "Image is null","Image Is Null!");
        }
    }
    else{
        QMessageBox::information(this, "No file selecte","No File Selecte!");
    }
}

void ImageShop::on_actionSave_S_triggered()
{
    if(this->image.data){
        QScreen *screen = QGuiApplication::primaryScreen();
        screen->grabWindow(ui->imageLabel->winId()).save(this->fileName);
        QMessageBox::information(this, "Save sucess","Save Sucess!");
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionSave_As_A_triggered()
{
    if(this->image.data){
        QString fileName = QFileDialog::getSaveFileName(this,"Save image","",tr("Image Files(*.png *.jpg  *.bmp *.jpeg)"));
        QScreen *screen = QGuiApplication::primaryScreen();
        screen->grabWindow(ui->imageLabel->winId()).save(fileName);
        QMessageBox::information(this, "Save sucess","Save Sucess!");
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionQuit_Q_triggered()
{
    QApplication::quit();
}

void ImageShop::on_actionRotate_90_triggered()
{
    if(this->image.data){
        double angle = 90;
        cv::Mat src = this->image;
        rotate(angle,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionRotate_180_triggered()
{
    if(this->image.data){
        double angle = 180;
        cv::Mat src = this->image;
        rotate(angle,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_rotate_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_rotate->setValue(value);
        double angle = value;
        cv::Mat src = this->image;
        rotate(angle,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_rotate_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_rotate->setValue(arg1);
        double angle = arg1;
        cv::Mat src = this->image;
        rotate(angle,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionUpDown_triggered()
{
    if(this->image.data){
        double flipCode = 0;
        cv::Mat src = this->image;
        flip(flipCode,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionLeftRight_triggered()
{
    if(this->image.data){
        double flipCode = 1;
        cv::Mat src = this->image;
        flip(flipCode,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionRectCut_toggled(bool arg1)
{
    if(arg1){
        ui->imageLabel->m_kind=1;
    }
    else{
        cv::Point start(ui->imageLabel->m_start.x(),ui->imageLabel->m_start.y());
        cv::Point stop(ui->imageLabel->m_stop.x(),ui->imageLabel->m_stop.y());
        cv::Mat src = this->image;
        cut(1,start,stop,src);
    }
}

void ImageShop::on_actionCircCut_toggled(bool arg1)
{
    if(arg1){
        ui->imageLabel->m_kind=2;
    }
    else{
        cv::Point start(ui->imageLabel->m_start.x(),ui->imageLabel->m_start.y());
        cv::Point stop(ui->imageLabel->m_stop.x(),ui->imageLabel->m_stop.y());
        cv::Mat src = this->image;
        cut(2,start,stop,src);
    }
}

void ImageShop::on_horizontalSlider_hue_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_hue->setValue(value);
        double hue = (double)value;
        double lightness = (double)ui->horizontalSlider_lightness->value()/100;
        double saturation = (double)ui->horizontalSlider_saturation->value()/100;
        cv::Mat src = this->image;
        colorChange(hue,lightness,saturation,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_hue_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_hue->setValue(arg1);
        double hue = (double)arg1;
        double lightness = (double)ui->horizontalSlider_lightness->value()/100;
        double saturation = (double)ui->horizontalSlider_saturation->value()/100;
        cv::Mat src = this->image;
        colorChange(hue,lightness,saturation,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_lightness_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_lightness->setValue(value);
        double lightness = (double)value/100;
        double hue = (double)ui->horizontalSlider_hue->value();
        double saturation = (double)ui->horizontalSlider_saturation->value()/100;
        cv::Mat src = this->image;
        colorChange(hue,lightness,saturation,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_lightness_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_lightness->setValue(arg1);
        double lightness = (double)arg1/100;
        double hue = (double)ui->horizontalSlider_hue->value();
        double saturation = (double)ui->horizontalSlider_saturation->value()/100;
        cv::Mat src = this->image;
        colorChange(hue,lightness,saturation,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_saturation_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_saturation->setValue(value);
        double saturation = (double)value/100;
        double hue = (double)ui->horizontalSlider_hue->value();
        double lightness = (double)ui->horizontalSlider_lightness->value()/100;
        cv::Mat src = this->image;
        colorChange(hue,lightness,saturation,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_saturation_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_saturation->setValue(arg1);
        double saturation = (double)arg1/100;
        double hue = (double)ui->horizontalSlider_hue->value();
        double lightness = (double)ui->horizontalSlider_lightness->value()/100;
        cv::Mat src = this->image;
        colorChange(hue,lightness,saturation,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionHalftone_triggered()
{
    if(this->image.data){
        cv::Mat src = this->image;
        colorHalftone(src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_transform_a_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_transform_a->setValue(value);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_transform_a_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_transform_a->setValue(arg1);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_transform_b_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_transform_b->setValue(value);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_transform_b_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_transform_b->setValue(arg1);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_transform_c_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_transform_c->setValue(value);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_transform_c_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_transform_c->setValue(arg1);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_transform_d_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_transform_d->setValue(value);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_transform_d_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_transform_d->setValue(arg1);
        int a = ui->horizontalSlider_transform_a->value();
        int b = ui->horizontalSlider_transform_b->value();
        int c = ui->horizontalSlider_transform_c->value();
        int d = ui->horizontalSlider_transform_d->value();
        cv::Mat src = this->image;
        segmentalLinearTransform(a,b,c,d,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_doubleSpinBox_logarithmic_transform_a_valueChanged(double arg1)
{
    if(this->image.data){
        double a = ui->doubleSpinBox_logarithmic_transform_a->value();
        double b = ui->doubleSpinBox_logarithmic_transform_b->value();
        double c = ui->doubleSpinBox_logarithmic_transform_c->value();
        cv::Mat src = this->image;
        logarithmicTransform(a,b,c,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_doubleSpinBox_logarithmic_transform_b_valueChanged(double arg1)
{
    if(this->image.data){
        double a = ui->doubleSpinBox_logarithmic_transform_a->value();
        double b = ui->doubleSpinBox_logarithmic_transform_b->value();
        double c = ui->doubleSpinBox_logarithmic_transform_c->value();
        cv::Mat src = this->image;
        logarithmicTransform(a,b,c,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_doubleSpinBox_logarithmic_transform_c_valueChanged(double arg1)
{
    if(this->image.data){
        double a = ui->doubleSpinBox_logarithmic_transform_a->value();
        double b = ui->doubleSpinBox_logarithmic_transform_b->value();
        double c = ui->doubleSpinBox_logarithmic_transform_c->value();
        cv::Mat src = this->image;
        logarithmicTransform(a,b,c,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_doubleSpinBox_exponential_trasform_a_valueChanged(double arg1)
{
    if(this->image.data){
        double a = ui->doubleSpinBox_exponential_trasform_a->value();
        double b = ui->doubleSpinBox_exponential_trasform_b->value();
        double c = ui->doubleSpinBox_exponential_trasform_c->value();
        cv::Mat src = this->image;
        exponentialTransform(a,b,c,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_doubleSpinBox_exponential_trasform_b_valueChanged(double arg1)
{
    if(this->image.data){
        double a = ui->doubleSpinBox_exponential_trasform_a->value();
        double b = ui->doubleSpinBox_exponential_trasform_b->value();
        double c = ui->doubleSpinBox_exponential_trasform_c->value();
        cv::Mat src = this->image;
        exponentialTransform(a,b,c,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_doubleSpinBox_exponential_trasform_c_valueChanged(double arg1)
{
    if(this->image.data){
        double a = ui->doubleSpinBox_exponential_trasform_a->value();
        double b = ui->doubleSpinBox_exponential_trasform_b->value();
        double c = ui->doubleSpinBox_exponential_trasform_c->value();
        cv::Mat src = this->image;
        exponentialTransform(a,b,c,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_doubleSpinBox_gamma_correct_valueChanged(double arg1)
{
    if(this->image.data){
        double r = ui->doubleSpinBox_gamma_correct->value();
        cv::Mat src = this->image;
        gammaCorrectTransform(r,src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionEqualization_triggered()
{
    if(this->image.data){
        cv::Mat src = this->image;
        Equalization(src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionSpecificationSML_toggled(bool arg1)
{

    if(arg1){
        QString fileName = QFileDialog::getOpenFileName(this,"Open file","","Image Files(*.jpg *.png *.bmp *.jpeg)");
        if(!fileName.isEmpty()){
            QTextCodec *code = QTextCodec::codecForName("gb18030");
            std::string name = code->fromUnicode(fileName).data();
            learn = cv::imread(name,CV_LOAD_IMAGE_UNCHANGED);
            cv::cvtColor(learn,learn,CV_BGR2RGB);
            if(learn.data){
                showImage(learn);
                getHistogram(learn);

            }
            else{
                QMessageBox::information(this, "Image is null","Image Is Null!");
            }
        }
        else{
            QMessageBox::information(this, "No file selecte","No File Selecte!");
        }
    }
    else{
        std::vector<double> learn_s = getAccumulateHistogram(learn);
        smlSpecification(learn_s);
    }
}

void ImageShop::on_actionSpecificationGML_toggled(bool arg1)
{
    if(arg1){
        QString fileName = QFileDialog::getOpenFileName(this,"Open file","","Image Files(*.jpg *.png *.bmp *.jpeg)");
        if(!fileName.isEmpty()){
            QTextCodec *code = QTextCodec::codecForName("gb18030");
            std::string name = code->fromUnicode(fileName).data();
            learn = cv::imread(name,CV_LOAD_IMAGE_UNCHANGED);
            cv::cvtColor(learn,learn,CV_BGR2RGB);
            if(learn.data){
                showImage(learn);
                getHistogram(learn);

            }
            else{
                QMessageBox::information(this, "Image is null","Image Is Null!");
            }
        }
        else{
            QMessageBox::information(this, "No file selecte","No File Selecte!");
        }
    }
    else{
        std::vector<double> learn_s = getAccumulateHistogram(learn);
        gmlSpecification(learn_s);
    }
}

void ImageShop::on_horizontalSlider_filter_valueChanged(int value)
{
    if(value % 2 ==1){
        ui->spinBox_filter->setValue(value);
    }
    else{
        ui->spinBox_filter->setValue(value+1);
        ui->horizontalSlider_filter->setValue(value+1);
    }
}

void ImageShop::on_spinBox_filter_valueChanged(int arg1)
{
    if(arg1 % 2 ==1){
        ui->horizontalSlider_filter->setValue(arg1);
    }
    else{
        ui->spinBox_filter->setValue(arg1+1);
        ui->horizontalSlider_filter->setValue(arg1+1);
    }
}

void ImageShop::on_actionMedianFilter_triggered()
{
    if(this->image.data){
        int size = ui->horizontalSlider_filter->value();
        cv::Mat src = this->image;
        medianFilter(src,size);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionGaussianFilter_triggered()
{
    if(this->image.data){
        int size = ui->horizontalSlider_filter->value();
        cv::Mat src = this->image;
        GaussianFilter(src,size);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionSharpen_triggered()
{
    if(this->image.data){
        cv::Mat src = this->image;
        sharpenFilter(src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionRaditionBlur_triggered()
{
    if(this->image.data){
        cv::Mat src = this->image;
        raditionBlur(src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionSnowBlur_triggered()
{
    if(this->image.data){
        cv::Mat src = this->image;
        snowBlur(src);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionBLPF_triggered()
{
    if(this->image.data){
        cv::Mat src = this->image;
        int n = ui->spinBox_BLPF_n->value();
        int D0 = ui->horizontalSlider_BLPF_D0->value();
        BLPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_BLPF_D0_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_BLPF_D0->setValue(value);
        cv::Mat src = this->image;
        int n = ui->spinBox_BLPF_n->value();
        int D0 = ui->horizontalSlider_BLPF_D0->value();
        BLPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_BLPF_D0_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_BLPF_D0->setValue(arg1);
        cv::Mat src = this->image;
        int n = ui->spinBox_BLPF_n->value();
        int D0 = ui->horizontalSlider_BLPF_D0->value();
        BLPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_BLPF_n_valueChanged(int arg1)
{
    if(this->image.data){
        cv::Mat src = this->image;
        int n = ui->spinBox_BLPF_n->value();
        int D0 = ui->horizontalSlider_BLPF_D0->value();
        BLPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_actionBHPF_triggered()
{
    if(this->image.data){
        cv::Mat src = this->image;
        int n = ui->spinBox_BHPF_n->value();
        int D0 = ui->horizontalSlider_BHPF_D0->value();
        BHPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_horizontalSlider_BHPF_D0_valueChanged(int value)
{
    if(this->image.data){
        ui->spinBox_BHPF_D0->setValue(value);
        cv::Mat src = this->image;
        int n = ui->spinBox_BHPF_n->value();
        int D0 = ui->horizontalSlider_BHPF_D0->value();
        BHPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_BHPF_D0_valueChanged(int arg1)
{
    if(this->image.data){
        ui->horizontalSlider_BHPF_D0->setValue(arg1);
        cv::Mat src = this->image;
        int n = ui->spinBox_BHPF_n->value();
        int D0 = ui->horizontalSlider_BHPF_D0->value();
        BHPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}

void ImageShop::on_spinBox_BHPF_n_valueChanged(int arg1)
{
    if(this->image.data){
        cv::Mat src = this->image;
        int n = ui->spinBox_BHPF_n->value();
        int D0 = ui->horizontalSlider_BHPF_D0->value();
        BHPF(src,n,D0);
    }
    else{
        QMessageBox::information(this, "Image is null","Image Is Null!");
    }
}
