import sys, os
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from PyQt5 import uic
from PyQt5.QtCore import *

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

#화면을 띄우는데 사용되는 Class 선언
class compareModel(QMainWindow, form_class1):
    command = QtCore.pyqtSignal(str) # 이미지 주소 전달
    model_1= ""
    model_2 = ""
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setStyleSheet("background-color: #F2F2F2;")
        

        # 위치 셋팅
        self.groupBox1.move(300,100)
        self.groupBox2.move(700,100)
        self.data_msg = QLabel("text", self)
        # self.groupBox1.addStrech(3)

        self.selectModel1.clicked.connect(self.choiceModel_1)
        self.selectModel2.clicked.connect(self.choiceModel_2)
        # 이미지 보여주기
        self.selectImage1.clicked.connect(self.folder_first)
        self.selectImage2.clicked.connect(self.folder_second)
        
        self.nextButton.clicked.connect(self.nextPage)

        #res 창

    def warningMSG(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.send_valve_popup_signal.emit(True)



    def folder_first(self):
        global working_path1
        # 폴더 구조 선택
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        working_path1 = QFileDialog.getExistingDirectory(self,"select Directory")
        # 비우고 경로 입력
        self.dir1.clear()
        self.dir1.append('경로: {}'.format(working_path1))


    def folder_second(self):
        global working_path2
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        working_path2 = QFileDialog.getExistingDirectory(self,"select Directory")
        self.dir2.clear()
        self.dir2.append('경로: {}'.format(working_path2))
        # filename = QFileDialog.getOpenFileName(self, 'Open File') 

    # model 1
    def choiceModel_1(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')[0]
        compareModel.model1 = name
        self.model_name1.clear()
        self.model_name1.append(name.split('/')[-1])
    # model 2 
    def choiceModel_2(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')[0]
        compareModel.model2 = name
        self.model_name2.clear()
        self.model_name2.append(name.split('/')[-1])

    def nextPage(self):
        if compareModel.compare_image2 and compareModel.compare_image1:
            resultModel(self)
        else:
            self.warningMSG("주의", "이미지와 모델을 먼저 집어넣어주세요.")




form_class2 = uic.loadUiType(form2)[0]
class resultModel(QMainWindow,form_class2):
    def __init__(self, parent):
        super(resultModel,self).__init__(parent)
        self.setupUi(self) # for_class2 ui 셋
        # UI 
        # print(parent.compare_image1)
        self.info1.append(parent.compare_image1[0])
        self.info2.append(parent.compare_image2[0])
        # uic.loadUi(form_class2,self)
        self.te = QTextEdit()
        self.lbl1 = QLabel('The number of words is 0')
        self.save.setStyleSheet('image:url(img/save.png);border:0px;')
        self.setGeometry(300, 300, 1000, 700)
        self.show()
    
 



# app = QApplication(sys.argv)
# new_widget = QStackedWidget()
# compare_model =compareModel()
# result_model =resultModel()
# new_widget.addWidget(compare_model)
# new_widget.addWidget(result_model)

# app.exec_()