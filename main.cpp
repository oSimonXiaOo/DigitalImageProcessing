#include "imageshop.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QFile qss(":/style/style.qss");
    qss.open(QFile::ReadOnly);
    a.setStyleSheet(qss.readAll());
    qss.close();

    ImageShop w;
    w.setWindowIcon(QIcon(":icon/Exe/ImageShop.png"));
    w.show();

    return a.exec();
}
