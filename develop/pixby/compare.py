import sys, os
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5 import uic

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
 
# 1. ui 연결 
# 연결할 ui 파일의 경로 설정
form1 = resource_path('ui/compare.ui')
form2 = resource_path('ui/res.ui')
# ui 로드 
form_class1 = uic.loadUiType(form1)[0]
form_class2 = uic.loadUiType(form2)[0]

#화면을 띄우는데 사용되는 Class 선언
class compareModel(QMainWindow, form_class1):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setStyleSheet("background-color: #777777;")
        self.comeImage.resize(250, 250)
        self.selectModel1.clicked.connect(self.fileopen)
        self.selectModel2.clicked.connect(self.fileopen)
        # 이미지 보여주기
        self.selectImage.clicked.connect(self.putImage)
        self.nextButton.clicked.connect(self.nextPage)

    def fileopen(self):
        global filename
        print('aa')
        filename = QFileDialog.getOpenFileName(self, 'Open File') 
        print(filename)

    def putImage(self):
        global compare_image
        # 이미지 주소 저장
        compare_image = QFileDialog.getOpenFileName(self, 'Open File')
        # 이미지를 라벨에 붙여서 뛰우기
        self.comeImage.setPixmap(QtGui.QPixmap(compare_image[0]))

    def nextPage(self):
        resWindow = resultModel()

        


class resultModel(QMainWindow, form_class2):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)


