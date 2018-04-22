#include "imagelabel.h"

ImageLabel::ImageLabel() :QLabel()
{
}

ImageLabel::ImageLabel(QWidget *parent) :QLabel()
{
}

ImageLabel::ImageLabel(int type) :QLabel()
{
    this->m_kind = type;
}

ImageLabel::~ImageLabel()
{
}

void ImageLabel::paintEvent(QPaintEvent*p)
{

    QLabel::paintEvent(p);//先调用父类的paintEvent为了显示'背景'!!!
    QPainter painter(this);
    QPen pen;       //设置画笔，颜色、宽度
    pen.setColor(Qt::white);
    pen.setWidth(2);
    painter.setPen(pen);

    if (m_kind == 1)//矩形
    {
        painter.drawRect(QRect(m_start,m_stop));
    }
    if (m_kind == 2){//画圆
        QPoint p;
        p.setX((m_start.x() + m_stop.x()) / 2);
        p.setY((m_start.y() + m_stop.y()) / 2);
        int radius = (int)sqrt((m_start.x() - m_stop.x())*(m_start.x() - m_stop.x()) + (m_start.y() - m_stop.y())*(m_start.y() - m_stop.y()))/2;
        painter.drawEllipse(p, radius, radius);
    }
}

void ImageLabel::mousePressEvent(QMouseEvent *e)
{
    if(e->button() && Qt::LeftButton){
        m_isDown = true;
        m_start = e->pos();
        m_stop = e->pos();
    }
}

void ImageLabel::mouseMoveEvent(QMouseEvent *e)
{
    if(m_isDown){
        m_stop = e->pos();
    }
    update();
}

void ImageLabel::mouseReleaseEvent(QMouseEvent *e)
{
    if(e->button() && Qt::LeftButton){
        m_isDown = false;
    }
    update();
}
