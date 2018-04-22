#-------------------------------------------------
#
# Project created by QtCreator 2017-11-26T14:55:49
#
#-------------------------------------------------

QT       += core gui
# RC_ICONS += imageshop.ico
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = DigitalImageProcessing
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH += $$(OPENCV_HOME)/include/

LIBS += $$(OPENCV_HOME)/x64/vc14/lib/opencv_world330.lib \
        $$(OPENCV_HOME)/x64/vc14/lib/opencv_world330d.lib


SOURCES += \
        main.cpp \
        imageshop.cpp \
        imagelabel.cpp

HEADERS += \
        imageshop.h \
        imagelabel.h

FORMS += \
        imageshop.ui

RESOURCES += \
    imageshop.qrc
