#ifndef IMAGESHOP_H
#define IMAGESHOP_H

#include <QMainWindow>
#include <QMessageBox>
#include <QFileDialog>
#include "imagelabel.h"
#include "opencv2/opencv.hpp"

namespace Ui {
class ImageShop;
}

class ImageShop : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageShop(QWidget *parent = 0);

    /*HomeworkOne*/
    void showImage(cv::Mat img);
    void rotate(double angle,cv::Mat src);
    void flip(int flipCode,cv::Mat src);
    void cut(int cutCode,cv::Point start,cv::Point stop,cv::Mat src);

    /*HomeworkTwo*/
    void RGB2HLS(double *h_hls, double *l_hls, double *s_hls, int r_rgb, int g_rgb, int b_rgb);
    void HLS2RGB(int *r_rgb, int *g_rgb, int *b_rgb, double h_hls, double l_hls, double s_hls);
    double HLS2RGBvalue(double h1,double h2, double hue);
    void colorChange(double hue,double lightness,double saturation,cv::Mat src);
    void colorHalftone(cv::Mat src);

    /*HomeworkThree*/
    void segmentalLinearTransform(int a,int b,int c,int d,cv::Mat src);
    void logarithmicTransform(double a,double b,double c,cv::Mat src);
    void exponentialTransform(double a,double b,double c,cv::Mat src);
    void gammaCorrectTransform(double r,cv::Mat src);
    void getHistogram(cv::Mat src);
    std::vector<double> getAccumulateHistogram(cv::Mat src);
    void Equalization(cv::Mat src);
    void gmlSpecification(std::vector<double> learn_s);
    void smlSpecification(std::vector<double> learn_s);

    /*HomeworkFour*/
    double **getGuassionArray(int size);
    void medianFilter (cv::Mat src,int size);
    void GaussianFilter(cv::Mat src,int size);
    void sharpenFilter(cv::Mat src);
    void raditionBlur(cv::Mat src);
    void snowBlur(cv::Mat src);

    /*HomeworkFive*/
    void BLPF(cv::Mat src,int N, int D0);
    void BHPF(cv::Mat src,int N, int D0);

    ~ImageShop();

private slots:
    /*HomeworkOne*/
    void on_actionOpen_O_triggered();

    void on_actionSave_S_triggered();

    void on_actionSave_As_A_triggered();

    void on_actionQuit_Q_triggered();

    void on_actionRotate_90_triggered();

    void on_actionRotate_180_triggered();

    void on_horizontalSlider_rotate_valueChanged(int value);

    void on_spinBox_rotate_valueChanged(int arg1);

    void on_actionUpDown_triggered();

    void on_actionLeftRight_triggered();

    void on_actionRectCut_toggled(bool arg1);

    void on_actionCircCut_toggled(bool arg1);

    /*HomeworkTwo*/
    void on_horizontalSlider_hue_valueChanged(int value);

    void on_spinBox_hue_valueChanged(int arg1);

    void on_horizontalSlider_lightness_valueChanged(int value);

    void on_spinBox_lightness_valueChanged(int arg1);

    void on_horizontalSlider_saturation_valueChanged(int value);

    void on_spinBox_saturation_valueChanged(int arg1);

    void on_actionHalftone_triggered();

    void on_horizontalSlider_transform_a_valueChanged(int value);

    void on_spinBox_transform_a_valueChanged(int arg1);

    void on_horizontalSlider_transform_b_valueChanged(int value);

    void on_spinBox_transform_b_valueChanged(int arg1);

    void on_horizontalSlider_transform_c_valueChanged(int value);

    void on_spinBox_transform_c_valueChanged(int arg1);

    void on_horizontalSlider_transform_d_valueChanged(int value);

    void on_spinBox_transform_d_valueChanged(int arg1);

    /*HomeworkThree*/
    void on_doubleSpinBox_logarithmic_transform_a_valueChanged(double arg1);

    void on_doubleSpinBox_logarithmic_transform_b_valueChanged(double arg1);

    void on_doubleSpinBox_logarithmic_transform_c_valueChanged(double arg1);

    void on_doubleSpinBox_exponential_trasform_a_valueChanged(double arg1);

    void on_doubleSpinBox_exponential_trasform_b_valueChanged(double arg1);

    void on_doubleSpinBox_exponential_trasform_c_valueChanged(double arg1);

    void on_doubleSpinBox_gamma_correct_valueChanged(double arg1);

    void on_actionEqualization_triggered();

    void on_actionSpecificationSML_toggled(bool arg1);

    void on_actionSpecificationGML_toggled(bool arg1);

    /*HomeworkFour*/
    void on_horizontalSlider_filter_valueChanged(int value);

    void on_spinBox_filter_valueChanged(int arg1);

    void on_actionMedianFilter_triggered();

    void on_actionGaussianFilter_triggered();

    void on_actionSharpen_triggered();

    void on_actionRaditionBlur_triggered();

    void on_actionSnowBlur_triggered();

    /*HomeworkFive*/
    void on_actionBLPF_triggered();

    void on_horizontalSlider_BLPF_D0_valueChanged(int value);

    void on_spinBox_BLPF_n_valueChanged(int arg1);

    void on_spinBox_BLPF_D0_valueChanged(int arg1);

    void on_actionBHPF_triggered();

    void on_horizontalSlider_BHPF_D0_valueChanged(int value);

    void on_spinBox_BHPF_n_valueChanged(int arg1);

    void on_spinBox_BHPF_D0_valueChanged(int arg1);

private:
    Ui::ImageShop *ui;
    QString fileName;
    cv::Mat image;

    cv::Mat learn;
};

#endif // IMAGESHOP_H
