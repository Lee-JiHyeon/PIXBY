import sys, os
import pixby.cnn.inferencing as Inf # cnn 추론 
import pixby.cnn.trainer as cnn_trainer
# from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
# from pixby.newSR import Create_SR_Model
# from pixby.compare import compareModel
from PyQt5 import QtGui, QtCore
from sqlite3.dbapi2 import connect

# ResNet_Base
# from pixby.cnn.ResNet_Base import

# from pixby.newSR import Create_SR_Model, Learn_SR_Model
# from pixby.compare import compareModel, resultModel
from pixby.srtest.src.please import Go

# from pixby.selectSR import Select_SR_Model
# form_class = uic.loadUiType("intro.ui")[0]

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# db연결

import sqlite3
class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('./pixby/ui/intro.ui', self)
        # self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        self.startBtn.clicked.connect(self.gotoChoice)
        self.compare.clicked.connect(self.gotoCompare)

    def gotoChoice(self):
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotoCompare(self):
        widget.setCurrentIndex(widget.currentIndex()+5)


class Choice(QMainWindow):
    def __init__(self):
        super(Choice, self).__init__()
        loadUi('./pixby/ui/choice.ui', self)

        self.gotoModelBtn.clicked.connect(self.gotoModel)
        self.goToCreateSR.clicked.connect(self.gotoCreateSR)
        backbutton = QPushButton(self)
        backbutton.move(0,10)
        backbutton.resize(80,80)
        backbutton.adjustSize()
        backbutton.setStyleSheet('image:url(img/undo.png);border:0px;background-color:#FFFFFF')
        backbutton.clicked.connect(self.goToBack)
    def goToBack(self):
        widget.setCurrentIndex(widget.currentIndex()-1)
    def gotoModel(self):
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotoCreateSR(self):
        widget.setCurrentIndex(widget.currentIndex()+2)


# ui 로드 
new_cnn_ui = resource_path('pixby/ui/newCNN.ui')
new_cnn_form = uic.loadUiType(new_cnn_ui)[0]
class Create_CNN_Model(QMainWindow, new_cnn_form):
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

    # 인자 값
    tr_path = ""
    te_path = ""
    sa_path = ""
    name = ""
    learning_rate=""
    epoch = ""
    batch = 32
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        self.save_btn.clicked.connect(self.save)
        self.train_btn.clicked.connect(self.train_folder)
        self.test_btn.clicked.connect(self.test_folder)
        self.start_button.clicked.connect(self.goToCNN)

        self.size32.setChecked(True)
        self.size16.clicked.connect(self.radioButtonClicked)
        self.size32.clicked.connect(self.radioButtonClicked)
        self.size64.clicked.connect(self.radioButtonClicked)

    def radioButtonClicked(self):
        if self.size16.isChecked():
            Create_CNN_Model.batch = 16
        elif self.size32.isChecked():
            Create_CNN_Model.batch = 32
        elif self.size64.isChecked():
            Create_CNN_Model.batch = 64
        print(Create_CNN_Model.batch)


    def warningMSG(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.send_valve_popup_signal.emit(True)

    def train_folder(self):
        # global working_path2
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        Create_CNN_Model.tr_path = QFileDialog.getExistingDirectory(self,"select Directory")
        self.train_path.clear()
        self.train_path.append('경로: {}'.format(Create_CNN_Model.tr_path))
  
    def test_folder(self):
        # global working_path2
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        Create_CNN_Model.te_path = QFileDialog.getExistingDirectory(self,"select Directory")
        self.test_path.clear()
        self.test_path.append('경로: {}'.format(Create_CNN_Model.te_path))

    # model 1
    def save(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.ShowDirsOnly
            Create_CNN_Model.sa_path = QFileDialog.getExistingDirectory(self,"select Directory")
            self.save_path.clear()
            self.save_path.append('경로: {}'.format(Create_CNN_Model.sa_path))
        except:
            self.warningMSG("주의", "경로 설정을 다시해주세요.")

    def goToCNN(self):
        Create_CNN_Model.learning_rate = self.learning_rate.toPlainText()
        Create_CNN_Model.epoch = self.epoch.toPlainText()
        Create_CNN_Model.name = self.name.toPlainText()
        Create_CNN_Model.learning_rate = float(Create_CNN_Model.learning_rate)
        # print(Create_CNN_Model.learning_rate)
        print("1")
        Create_CNN_Model.epoch = int(Create_CNN_Model.epoch)
        print("2")
        if Create_CNN_Model.learning_rate >=0.1:
            self.warningMSG("주의", "learning rate는 0.1보다 작게 해주셔야 됩니다.")
        elif Create_CNN_Model.sa_path and Create_CNN_Model.tr_path and Create_CNN_Model.te_path:
            print(Create_CNN_Model.tr_path, Create_CNN_Model.te_path,  Create_CNN_Model.sa_path, Create_CNN_Model.name ,Create_CNN_Model.learning_rate, Create_CNN_Model.epoch, Create_CNN_Model.batch)
            cnn_trainer.CNN_Train(Create_CNN_Model.tr_path, Create_CNN_Model.te_path,  Create_CNN_Model.sa_path, Create_CNN_Model.name ,Create_CNN_Model.learning_rate, Create_CNN_Model.epoch, Create_CNN_Model.batch)
        else:
            self.warningMSG("주의", "이미지와 모델을 먼저 집어넣어주세요.")
        # except:
            # self.warningMSG("주의", "LEARNING RATE, epoch을 숫자로 입력해주셔야 합니다.")

        


compare_ui = resource_path('pixby/ui/compare.ui')
compare_form = uic.loadUiType(compare_ui)[0]
#화면을 띄우는데 사용되는 Class 선언
class Compare_Model(QMainWindow, compare_form ):
    command = QtCore.pyqtSignal(str) # 이미지 주소 전달
    model_1= ""
    model_2 = ""
    working_path1 = ""
    working_path2 = ""
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
        
        # backbutton 
        backbutton = QPushButton(self)
        backbutton.move(0,10)
        backbutton.resize(80,80)
        backbutton.adjustSize()
        backbutton.setStyleSheet('image:url(img/undo.png);border:0px;background-color:#F2F2F2')
        backbutton.clicked.connect(self.goToBack)
    

    
        self.selectModel1.clicked.connect(self.choiceModel_1)
        self.selectModel2.clicked.connect(self.choiceModel_2)
        # 이미지 보여주기
        self.selectImage1.clicked.connect(self.folder_first)
        self.selectImage2.clicked.connect(self.folder_second)
        
        self.nextButton.clicked.connect(self.nextPage)



    # 뒤로가기 -> classfication 설정 페이지
    def goToBack(self):
        widget.setCurrentIndex(widget.currentIndex()+1)

        # widget.setCurrentIndex(widget.currentIndex()-4)
    #경고 창 메서드...
    def warningMSG(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.send_valve_popup_signal.emit(True)



    def folder_first(self):
        # global working_path1
        # 폴더 구조 선택
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        Compare_Model.working_path1 = QFileDialog.getExistingDirectory(self,"select Directory")
        # 비우고 경로 입력
        self.dir1.clear()
        self.dir1.append('경로: {}'.format(Compare_Model.working_path1))


    def folder_second(self):
        # global working_path2
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        Compare_Model.working_path2 = QFileDialog.getExistingDirectory(self,"select Directory")
        self.dir2.clear()
        self.dir2.append('경로: {}'.format(Compare_Model.working_path2))

    # model 1
    def choiceModel_1(self):
        try:
            name = QFileDialog.getOpenFileName(self, 'Open File')[0]
            Compare_Model.model_1 = name # 분류모델 경로
            self.model_name1.clear()
            self.model_name1.append(name.split('/')[-1])
        except:
            self.warningMSG("주의", "모델 파일이 아닙니다.")
    # model 2 
    def choiceModel_2(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')[0]
        Compare_Model.model_2 = name

        # 모델 경로 보여주기
        self.model_name2.clear()
        self.model_name2.append(name.split('/')[-1])

    def nextPage(self):
        if Compare_Model.working_path1 and Compare_Model.working_path2:
            Inf.Infer(Compare_Model.working_path1, Compare_Model.model_1)
            Inf.Infer(Compare_Model.working_path2, Compare_Model.model_2)
            Result_Model(self)
        else:
            self.warningMSG("주의", "이미지와 모델을 먼저 집어넣어주세요.")



# SR 이전 모델 선택 
class Select_SR_Model(QMainWindow):

    def __init__(self):
        super(Select_SR_Model, self).__init__()
        loadUi('./pixby/ui/select.ui', self)

        # sql 연동
        self.sqlConnect()

        # 이미지 열기 버튼
        self.imageBtn.clicked.connect(self.openImage)
        backbutton = QPushButton(self)
        backbutton.move(0,10)
        backbutton.resize(80,80)
        backbutton.adjustSize()
        backbutton.setStyleSheet('image:url(img/undo.png);border:0px;background-color:#F2F2F2')
        backbutton.clicked.connect(self.goToBack)

    def goToBack(self):
        widget.setCurrentIndex(widget.currentIndex()-1)
    def openImage(self):
        imageOpen = QFileDialog.getOpenFileName(self, 'open file', './')

    # DB) SQL 연결 및 테이블 생성
    def sqlConnect(self):
        # db파일 이름 설정
        self.dbName = "db.sqlite3"

        self.conn = sqlite3.connect(self.dbName, isolation_level=None)

        # 커서 객체를 받아와서 execute 메서드로 CREATE TABLE 쿼리를 전송합니다.
        self.cur = self.conn.cursor()
        # 모델테이블 생성 임시로test라고 해놓음(모델이름있으면 오류남)
        # self.cur.execute("CREATE TABLE Test(Model Name, Epoch);")
        # 테이블에 값 넣기
        self.cur.execute("INSERT INTO Test Values('ThirdSR', '8');")

        # pyqt창에 표로 db데이터 보여주기 함수실행
        self.getData()

    # db값 가져오기 + 표로 보여주기
    def getData(self):
        connection = sqlite3.connect("db.sqlite3")
        self.cur = connection.cursor()
        sqlquery = 'SELECT * FROM Test'

        # 표 행값구하기위해 fetchall로 [(), ()..] 형태 만들어줌
        data_list = self.cur.execute(sqlquery).fetchall()

        # 보여줄 행 갯수 설정
        self.tableWidget.setRowCount(len(data_list))
        # 행 전체 클릭하게 하는 코드
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        # 클릭한 행의 모델값 전달함수
        self.tableWidget.cellClicked.connect(self.selectSR)

        tablerow = 0
        for row in self.cur.execute(sqlquery):
            # print(row)
            # print(row[0])
            self.tableWidget.setItem(tablerow, 0, QTableWidgetItem(row[0]))
            self.tableWidget.setItem(tablerow, 1, QTableWidgetItem(row[1]))
            tablerow += 1

    def selectSR(self, row):
        # pyqt창 선택한 모델이름 표시
        data = self.tableWidget.item(row, 0)
        select_sr_modelname = data.text()
        self.selectSRModelName.setText(select_sr_modelname)




# 1. ui 연결
# 연결할 ui 파일의 경로 설정
new_sr_ui = resource_path('pixby/ui/newSR.ui')
learn_ui = resource_path('pixby/ui/learn.ui')
# ui 로드 
new_sr_form  = uic.loadUiType(new_sr_ui )[0]
learn_ui_form  = uic.loadUiType(learn_ui )[0]


class Thread1(QThread):
    #parent = MainWidget을 상속 받음.
    def __init__(self, parent=None):
        super().__init__(parent)
        self.threadpool = QThreadPool()

    def run(self):
        Go()

#화면을 띄우는데 사용되는 Class 선언
class Create_SR_Model(QMainWindow, new_sr_form) :
    filename = ''

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.dataLoadFn)
        backbutton = QPushButton(self)
        backbutton.move(0,10)
        backbutton.resize(80,80)
        backbutton.adjustSize()
        backbutton.setStyleSheet('image:url(img/undo.png);border:0px;background-color:#F2F2F2')
        backbutton.clicked.connect(self.goToBack)

    def goToBack(self):
        widget.setCurrentIndex(widget.currentIndex()-2)
    def dataLoadFn(self):
        x = Thread1(self)
        x.start()
        # Go()
        # filename = QFileDialog.getOpenFileName(self, 'Open File', './')
        # print(filename)
        # if filename:
        #     self.label_34.setPixmap(QtGui.QPixmap("filename"))
        #     self.label_34.setGeometry(QtCore.QRect(100, 100))



class Learn_SR_Model(QMainWindow, learn_ui_form) :
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




res_ui = resource_path('pixby/ui/res.ui')
res_form = uic.loadUiType(res_ui)[0]
class Result_Model(QMainWindow,res_form):
    def __init__(self, parent):
        super(Result_Model,self).__init__(parent)
        self.setupUi(self) # for_class2 ui 셋
        # UI 
        print(parent.working_path1)
        # 모델 경로 출력
        self.info1.append(parent.working_path1)
        self.info2.append(parent.working_path2)
        # uic.loadUi(form_class2,self)
        self.te = QTextEdit()
        self.lbl1 = QLabel('The number of words is 0')
        self.save.setStyleSheet('image:url(img/save.png);border:0px;')
        self.setGeometry(300, 300, 1000, 700)
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = QStackedWidget()
    windowclass = WindowClass()
    choice = Choice()
    select_sr_model = Select_SR_Model()
    create_sr = Create_SR_Model()
    learn_sr = Learn_SR_Model()
    compare_model = Compare_Model()
    create_cnn = Create_CNN_Model()
    widget.addWidget(windowclass)
    widget.addWidget(choice)
    widget.addWidget(select_sr_model)
    widget.addWidget(create_sr)
    widget.addWidget(learn_sr)
    widget.addWidget(compare_model)
    widget.addWidget(create_cnn)
    widget.setFixedHeight(960)
    widget.setFixedWidth(1280)
    widget.show()
    app.exec_()
