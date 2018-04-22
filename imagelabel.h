#ifndef IMAGELABEL_H
#define IMAGELABEL_H

#include <QtWidgets>

class ImageLabel : public QLabel
{
public:
    explicit ImageLabel();
    explicit ImageLabel(QWidget *parent = 0);
    explicit ImageLabel(int type);
    ~ImageLabel();

    void paintEvent(QPaintEvent* p);
    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);

    QPoint m_start;
    QPoint m_stop;
    bool m_isDown;
    int m_kind;
};


#endif // IMAGELABEL_H
