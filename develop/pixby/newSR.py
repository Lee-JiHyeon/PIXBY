import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtGui, QtCore
from pixby.srtest.src.please import Go

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
# form_class = uic.loadUiType("ui/newSR.ui")[0]

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
 
# 1. ui 연결 
# 연결할 ui 파일의 경로 설정
form1 = resource_path('ui/newSR.ui')
form2 = resource_path('ui/learn.ui')
# ui 로드 
form_class1 = uic.loadUiType(form1)[0]
form_class2 = uic.loadUiType(form2)[0]


class Thread1(QThread):
    #parent = MainWidget을 상속 받음.
    def __init__(self, parent=None):
        super().__init__(parent)
    def run(self):
        Go()

#화면을 띄우는데 사용되는 Class 선언
class Create_SR_Model(QMainWindow, form_class1) :
    filename = ''

    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.dataLoadFn)

    
    def dataLoadFn(self):
<<<<<<< HEAD
        filename = QFileDialog.getOpenFileName(self, 'Open File', './')
        print(filename)
        if filename:
            self.label_34.setPixmap(QtGui.QPixmap("filename"))
            self.label_34.setGeometry(QtCore.QRect(100, 100, width_size, height_size))
=======
        x = Thread1(self)
        x.start()
        # Go()
        # filename = QFileDialog.getOpenFileName(self, 'Open File', './')
        # print(filename)
        # if filename:
        #     self.label_34.setPixmap(QtGui.QPixmap("filename"))
        #     self.label_34.setGeometry(QtCore.QRect(100, 100))



class Learn_SR_Model(QMainWindow, form_class2) :
    filename = ''

    def __init__(self) :
        super().__init__()
        self.setupUi(self)

    #     self.pushButton.clicked.connect(self.dataLoadFn)

    
    # def dataLoadFn(self):
    #     filename = QFileDialog.getOpenFileName(self, 'Open File', './')
    #     print(filename)
    #     if filename:
    #         self.label_34.setPixmap(QtGui.QPixmap("filename"))
    #         self.label_34.setGeometry(QtCore.QRect(100, 100, width_size, height_size))
>>>>>>> f05acce23935333602591b6b31db3ef6fcab1056
