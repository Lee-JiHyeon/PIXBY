import sys
import os
import pixby.cnn.inferencing as Inf  # cnn 추론
import pixby.cnn.trainer as cnn_trainer
import sqlite3
# from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import glob
from PIL import Image
import shutil

# from pixby.newSR import Create_SR_Model
# from pixby.compare import compareModel
from PyQt5 import QtGui, QtCore
from sqlite3.dbapi2 import connect
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ResNet_Base
# from pixby.cnn.ResNet_Base import

# from pixby.newSR import Create_SR_Model, Learn_SR_Model
# from pixby.compare import compareModel, resultModel
from pixby.srtest.src.main import main

# from pixby.selectSR import Select_SR_Model
# form_class = uic.loadUiType("intro.ui")[0]

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

os.makedirs('./SRimages', exist_ok=True)
os.makedirs('./SRimages/TESTDATA', exist_ok=True)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(
            os.path.abspath(__file__)))
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# db연결


class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('./pixby/ui/intro.ui', self)
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())
        self.startBtn.clicked.connect(self.gotoChoice)

    def gotoChoice(self):
        widget.setCurrentWidget(choice)


class Choice(QMainWindow):
    def __init__(self):
        super(Choice, self).__init__()
        loadUi('./pixby/ui/choice.ui', self)

        self.goToCreateSR.clicked.connect(self.gotoCreateSR)
        self.goToSelectSR.clicked.connect(self.gotoResultSR)
        self.goToCreateCNN.clicked.connect(self.gotoCreateCNN)
        self.goToCompare.clicked.connect(self.gotoCompare)

        backbutton = QPushButton(self)
        backbutton.move(0, 10)
        backbutton.resize(80, 80)
        backbutton.adjustSize()
        backbutton.setStyleSheet(
            'image:url(img/undo.png);border:0px;background-color:#e7e6e1')
        backbutton.clicked.connect(self.goToBack)

    def goToBack(self):
        widget.setCurrentWidget(windowclass)

    def gotoResultSR(self):
        widget.setCurrentWidget(result_sr)

    def gotoCreateSR(self):
        widget.setCurrentWidget(create_sr)

    def gotoCreateCNN(self):
        widget.setCurrentWidget(create_cnn)

    def gotoCompare(self):
        widget.setCurrentWidget(compare_model)


# cnn thread
class Thread3(QThread):
    # parent = MainWidget을 상속 받음.
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.threadpool = QThreadPool()
        # 학습 시작
        # print(parent.tr_path, parent.te_path, parent.sa_path, parent.model_name,parent.lr, parent.ep, parent.batch)

    def run(self):
        cnn_trainer.CNN_Train(
            train_cnn, cnn_data[0], cnn_data[1], cnn_data[2], cnn_data[3], cnn_data[4], cnn_data[5], cnn_data[6])


# ui 로드
new_cnn_ui = resource_path('pixby/ui/newCNN.ui')
new_cnn_form = uic.loadUiType(new_cnn_ui)[0]
cnn_data = [0, 0, 0, 0, 0, 0, 32]


class Create_CNN_Model(QMainWindow, new_cnn_form):
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')
    # 인자 값
    tr_path = ""
    te_path = ""
    sa_path = ""
    model_name = ""
    lr = ""
    ep = ""
    batch = 32

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.train_dir.doubleClicked.connect(self.showImg1)
        self.test_dir.doubleClicked.connect(self.showImg2)
        self.save_btn.clicked.connect(self.save)
        self.train_btn.clicked.connect(self.train_folder)
        self.test_btn.clicked.connect(self.test_folder)
        self.start_button.clicked.connect(self.goToCNN)

        self.size32.setChecked(True)
        self.size16.clicked.connect(self.radioButtonClicked)
        self.size32.clicked.connect(self.radioButtonClicked)
        self.size64.clicked.connect(self.radioButtonClicked)

        backbutton = QPushButton(self)
        backbutton.move(0, 10)
        backbutton.resize(80, 80)
        backbutton.adjustSize()
        backbutton.setStyleSheet(
            'image:url(img/undo.png);border:0px;background-color:#e7e6e1')
        backbutton.clicked.connect(self.goToBack)

        backbutton2 = QPushButton(self)
        backbutton2.move(60, 10)
        backbutton2.resize(150, 150)
        backbutton2.adjustSize()
        backbutton2.setStyleSheet(
            'image:url(img/house.png);border:0px;background-color:#e7e6e1')
        backbutton2.clicked.connect(self.goToHome)

        speaker = QLabel(self)
        speaker.move(10,900)
        speaker.resize(30,30)
        speaker.setStyleSheet(
            'image:url(img/megaphone.png);border:0px;background-color:#f7f6e7')
    def goToHome(self):
        widget.setCurrentWidget(choice)


    def goToBack(self):
        widget.setCurrentWidget(choice)

    def radioButtonClicked(self):
        if self.size16.isChecked():
            Create_CNN_Model.batch = 16
        elif self.size32.isChecked():
            Create_CNN_Model.batch = 32
        elif self.size64.isChecked():
            Create_CNN_Model.batch = 64
        print(Create_CNN_Model.batch)

    def showImg1(self, index):
        self.mainImg = self.train_dir.model().filePath(index)
        pixmap = QtGui.QPixmap(self.mainImg)
        self.image_view.setPixmap(pixmap)
        pixmap = pixmap.scaled(340, 350, Qt.IgnoreAspectRatio)
        self.image_view.setPixmap(pixmap)
        img = Image.open(self.mainImg)
        # st = os.stat(self.mainImg)
        self.file_name.setText(img.filename.split('/')[-1])
        self.file_size.setText("{} X {}".format(
            str(img.width), str(img.height)))
        self.fomat.setText(img.format)
        self.class_name.setText(img.filename.split('/')[-2])

        # self.image_view.adjustSize()

    def showImg2(self, index):
        self.mainImg = self.test_dir.model().filePath(index)
        pixmap = QtGui.QPixmap(self.mainImg)
        pixmap = pixmap.scaled(340, 350, Qt.IgnoreAspectRatio)
        self.image_view.setPixmap(pixmap)
        # self.image_view.adjustSize()
        img = Image.open(self.mainImg)
        st = os.stat(self.mainImg)
        self.file_name.setText(img.filename.split('/')[-1])
        self.file_size.setText("{} X {}".format(
            str(img.width), str(img.height)))
        self.fomat.setText(img.format)
        self.class_name.setText(img.filename.split('/')[-2])

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
        Create_CNN_Model.tr_path = QFileDialog.getExistingDirectory(
            self, "select Directory")
        self.train_path.clear()
        self.train_path.append('{}'.format(
            Create_CNN_Model.tr_path.split('/')[-1]))
        treeModel = QFileSystemModel()
        self.train_dir.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.train_dir.setRootIndex(treeModel.index(Create_CNN_Model.tr_path))
        self.train_dir.hideColumn(1)
        self.train_dir.hideColumn(2)
        self.train_dir.hideColumn(3)

    def test_folder(self):
        # global working_path2
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        Create_CNN_Model.te_path = QFileDialog.getExistingDirectory(
            self, "select Directory")
        self.test_path.clear()
        self.test_path.append('{}'.format(
            Create_CNN_Model.te_path.split('/')[-1]))
        treeModel = QFileSystemModel()
        self.test_dir.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.test_dir.setRootIndex(treeModel.index(Create_CNN_Model.te_path))
        self.test_dir.hideColumn(1)
        self.test_dir.hideColumn(2)
        self.test_dir.hideColumn(3)

    # model 1
    def save(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.ShowDirsOnly
            Create_CNN_Model.sa_path = QFileDialog.getExistingDirectory(
                self, "select Directory")
            self.save_path.clear()
            self.save_path.append('{}'.format(Create_CNN_Model.sa_path))
        except:
            self.warningMSG("주의", "경로 설정을 다시해주세요.")

    # cnn 학습 페이지 이동
    def goToCNN(self):
        global cnn_data
        Create_CNN_Model.lr = self.learning_rate.toPlainText()
        Create_CNN_Model.ep = self.epoch.toPlainText()
        Create_CNN_Model.model_name = self.name.toPlainText()
        try:
            Create_CNN_Model.lr = float(Create_CNN_Model.lr)
            Create_CNN_Model.ep = int(Create_CNN_Model.ep)
            cnn_data = [Create_CNN_Model.tr_path, Create_CNN_Model.te_path, Create_CNN_Model.sa_path,
                        Create_CNN_Model.model_name, Create_CNN_Model.lr, Create_CNN_Model.ep, Create_CNN_Model.batch]
            # print(Create_CNN_Model.learning_rate)

            if Create_CNN_Model.lr >= 0.1:
                self.warningMSG("주의", "learning rate는 0.1보다 작게 해주셔야 됩니다.")
            elif Create_CNN_Model.sa_path and Create_CNN_Model.tr_path and Create_CNN_Model.te_path:
                widget.setCurrentWidget(train_cnn)
                train_cnn.model_name.setText(Create_CNN_Model.model_name)
                train_cnn.batch_size.setText(str(Create_CNN_Model.batch))
                train_cnn.lr_rate.setText(str(Create_CNN_Model.lr))
                train_cnn.epoch.setText(str(Create_CNN_Model.ep))
                # widget.setCurrentWidget(widget.currentIndex()+1)
            else:
                self.warningMSG("주의", "경로를 집어넣어주세요.")
        except:
            self.warningMSG("주의", "필요한 값을 모두 넣어주세요.")


# cnn 학습 뷰
train_cnn_ui = resource_path('pixby/ui/trainCNN.ui')
train_form = uic.loadUiType(train_cnn_ui)[0]


class Train_CNN(QMainWindow, train_form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # print(self.cnn_train.name)
        self.gotestbutton.clicked.connect(self.test)
        backbutton = QPushButton(self)
        backbutton.move(0, 10)

        backbutton.resize(80, 80)
        backbutton.adjustSize()

        backbutton.setStyleSheet(
            'image:url(img/undo.png);border:0px;background-color:#e7e6e1')
        backbutton.clicked.connect(self.goToBack)
        self.fig1 = plt.Figure()
        self.canvas1 = FigureCanvas(self.fig1)
        self.fig2 = plt.Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        self.layout1.addWidget(self.canvas1)
        self.layout2.addWidget(self.canvas2)
        self.gotoInf.clicked.connect(self.goToNext)
        speaker = QLabel(self)
        speaker.move(10,900)
        speaker.resize(30,30)
        speaker.setStyleSheet(
            'image:url(img/megaphone.png);border:0px;background-color:#f7f6e7')
    def goToBack(self):
        widget.setCurrentWidget(create_cnn)

    def goToNext(self):
        widget.setCurrentWidget(compare_model)

    def test(self):
        t = Thread3(self)
        t.start()


compare_ui = resource_path('pixby/ui/compare.ui')
compare_form = uic.loadUiType(compare_ui)[0]
# 화면을 띄우는데 사용되는 Class 선언





# cnn 추론
class Thread4(QThread):
    # parent = MainWidget을 상속 받음.
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.Mname = parent.m_name
        # self.threadpool = QThreadPool()
        self.path1 = parent.datas[2]
        self.path2 = parent.datas[3]
        self.model1 = parent.datas[0]
        self.model2 = parent.datas[1]
        self.compare_data = parent
    def run(self):
        global res1, res2 
        self.res1_loss, self.res1_accuracy, self.res1_matrix = Inf.Infer(self.path1, self.model1 )
        self.res2_loss, self.res2_accuracy, self.res2_matrix = Inf.Infer(self.path2, self.model2 )
        compare_model.nextButton.setEnabled(True)
        self.res1_loss, self.res1_accuracy = round(
        self.res1_loss, 4), round(self.res1_accuracy, 2)
        self.res2_loss, self.res2_accuracy = round(
            self.res2_loss, 4), round(self.res2_accuracy, 2)
        self.compare_data.compare_table.setItem(0,0, QTableWidgetItem(str(self.res1_accuracy)))
        self.compare_data.compare_table.setItem(0, 1, QTableWidgetItem(str(self.res1_loss)))
        self.compare_data.compare_table.setItem(
            1, 0, QTableWidgetItem(str(self.res2_accuracy)))
        self.compare_data.compare_table.setItem(1, 1, QTableWidgetItem(str(self.res2_loss)))

        length = len(self.res1_matrix)
        self.compare_data.model1.setRowCount(length)
        self.compare_data.model2.setRowCount(length)
        self.compare_data.model1.setColumnCount(length)
        self.compare_data.model2.setColumnCount(length)

        for i in range(length):
            for j in range(length):
                self.compare_data.model1.setItem(i, j, QTableWidgetItem(
                    str(self.res1_matrix[i][j])))

        for i in range(length):
            for j in range(length):
                self.compare_data.model2.setItem(i, j, QTableWidgetItem(
                    str(self.res2_matrix[i][j])))

class Compare_Model(QMainWindow, compare_form):
    command = QtCore.pyqtSignal(str)  # 이미지 주소 전달
    model_1 = ""
    model_2 = ""
    working_path1 = ""
    working_path2 = ""
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.setStyleSheet("background-color: #F2F2F2;")

        # 위치 셋팅
        self.data_msg = QLabel("text", self)
        # self.groupBox1.addStrech(3)

        # backbutton
        backbutton = QPushButton(self)
        backbutton.move(0, 10)
        backbutton.resize(80, 80)
        backbutton.adjustSize()
        backbutton.setStyleSheet(
            'image:url(img/undo.png);border:0px;background-color:#e7e6e1')
        backbutton.clicked.connect(self.goToBack)

        backbutton2 = QPushButton(self)
        backbutton2.move(60, 10)
        backbutton2.resize(150, 150)
        backbutton2.adjustSize()
        backbutton2.setStyleSheet(
            'image:url(img/house.png);border:0px;background-color:#e7e6e1')
        backbutton2.clicked.connect(self.goToHome)

        speaker = QLabel(self)
        speaker.move(10,890)
        speaker.resize(30,30)
        speaker.setStyleSheet(
            'image:url(img/megaphone.png);border:0px;background-color:#f7f6e7')
        
        self.selectModel1.clicked.connect(self.choiceModel_1)
        self.selectModel2.clicked.connect(self.choiceModel_2)
        # 이미지 보여주기
        self.selectImage1.clicked.connect(self.folder_first)
        self.selectImage2.clicked.connect(self.folder_second)

        self.nextButton.clicked.connect(self.nextPage)

    # 뒤로가기 -> classfication 설정 페이지


    # 뒤로가기
    def goToBack(self):
        widget.setCurrentWidget(create_cnn)
    #홈버튼
    def goToHome(self):
        widget.setCurrentWidget(choice)
    # 경고문 함수 
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
        Compare_Model.working_path1 = QFileDialog.getExistingDirectory(
            self, "select Directory")
        # 비우고 경로 입력
        self.dir1.clear()
        self.dir1.append('경로: {}'.format(Compare_Model.working_path1))

    def folder_second(self):
        # global working_path2
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        Compare_Model.working_path2 = QFileDialog.getExistingDirectory(
            self, "select Directory")
        self.dir2.clear()
        self.dir2.append('경로: {}'.format(Compare_Model.working_path2))

    # model 1
    def choiceModel_1(self):
        try:
            name = QFileDialog.getOpenFileName(self, 'Open File')[0]
            Compare_Model.model_1 = name  # 분류모델 경로
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
        self.nextButton.setEnabled(False)
        if Compare_Model.working_path1 and Compare_Model.working_path2:
            self.res1 = Inf.Infer(
                Compare_Model.working_path1, Compare_Model.model_1)
            self.res2 = Inf.Infer(
                Compare_Model.working_path2, Compare_Model.model_2)
            Result_Model(self)

        else:
            self.warningMSG("주의", "이미지와 모델을 먼저 집어넣어주세요.")

# 1. ui 연결
# 연결할 ui 파일의 경로 설정
new_sr_ui = resource_path('pixby/ui/newSR.ui')
learn_ui = resource_path('pixby/ui/learn.ui')
# ui 로드
new_sr_form = uic.loadUiType(new_sr_ui)[0]
learn_ui_form = uic.loadUiType(learn_ui)[0]


create_sr_data = {
    'model_name': 'asad',
    'filename': '',
    'batch_size': 16,
    'learning_rate': 0.0001,
    'epoch': 5,
    'resblock': 16,
    'features': 32,
    'scale': 'x2',
    'data_dir': '',
    'save_dir': ''
}

learning = {
    'model': 'EDSR',
    'scale': [2],
    'save': 'test',
    'pre_train': './pixby/srtest/experiment/edsr_baseline_x2/model/model_best.pt',
    'chop': True,
    'dir_data': './SRimages',
    'data_train': ['TESTDATA'],
    'data_test': ['TESTDATA'],
    'data_range': '1-8/9-10',
    'epochs': 2,
    'ext': 'img',
    'save_results': True,
    'batch_size': 4,  # default 16
    'lr': 0.0001,  # default 1e-4
    'n_resblocks': 16,
    'n_feats': 64,
    # 'n_threads' : 0
}
#  전이학습을 위한 데이터

testing = {
    'data_test':  ['Demo'],
    'test_only': True,
    'save_results': True,
    'chop': True,
    # 'n_threads' : 0,
    'scale': [2],
    'n_resblocks': 16,
    'n_feats': 64,
    'pre_train': './pixby/srtest/experiment/edsr_baseline_x2/model/model_best.pt',
    'save': 'SRresults',
    'dir_demo': './SRimages'

}


#  sr을 하기위한 실제 이미지

class Thread1(QThread):
    # parent = MainWidget을 상속 받음.

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.threadpool = QThreadPool()
        # print(parent.m_name)
        self.textBox_terminal = parent.textBox_terminal

    def run(self):
        main(learn_sr, **learning)
        self.textBox_terminal.append('학습이 종료되었습니다.')


class Thread2(QThread):
    # parent = MainWidget을 상속 받음.
    def __init__(self, parent=None):
        super().__init__(parent)

        self.textBox_terminal = parent.textBox_terminal
        # self.Mname = parent.m_name
        # self.threadpool = QThreadPool()

    def run(self):
        main(result_sr, **testing)
        # Result_SR_Model.setResImg(Result_SR_Model())

        self.textBox_terminal.append('SR 과정이 끝났습니다.')
        self.textBox_terminal.append('결과보기 버튼을 클릭하여 결과를 확인하세요.')
        
        


# 화면을 띄우는데 사용되는 Class 선언


class Create_SR_Model(QMainWindow, new_sr_form):
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

   # 보낼 시그널 데이터 타입 , 갯수 지정
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.dataLoadFn)
        backbutton = QPushButton(self)
        backbutton.move(0, 10)
        backbutton.resize(80, 80)
        backbutton.adjustSize()
        backbutton.setStyleSheet(
            'image:url(img/undo.png);border:0px;background-color:#e7e6e1')
        backbutton.clicked.connect(self.goToBack)

        speaker = QLabel(self)
        speaker.move(10,900)
        speaker.resize(30,30)
        speaker.setStyleSheet(
            'image:url(img/megaphone.png);border:0px;background-color:#f7f6e7')
        self.traindataButton.clicked.connect(self.data_dir_save)

        # batch_box = self.batchtextEdit
        self.batchtextEdit.textChanged.connect(self.batch_changed)
        self.learningtextEdit.textChanged.connect(self.learning_changed)
        self.epochtextEdit.textChanged.connect(self.epoch_changed)
        self.modelnametextEdit.textChanged.connect(self.model_name_changed)

        res_box = self.resblockcombobox
        res_box.addItem('16')
        res_box.addItem('32')
        res_box.addItem('48')
        # res_box.addItem('64')

        feature_box = self.featurecombobox
        feature_box.addItem('32')
        feature_box.addItem('64')
        scale_box = self.scalecombobox
        scale_box.addItem('x2')
        scale_box.addItem('x3')
        scale_box.addItem('x4')

        res_box.activated[str].connect(self.onRes)
        feature_box.activated[str].connect(self.onFeature)
        scale_box.activated[str].connect(self.onScale)

        self.treeView.doubleClicked.connect(self.showImg)

    def warningMSG(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.send_valve_popup_signal.emit(True)

    def data_dir_save(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        create_sr_data['data_dir'] = QFileDialog.getExistingDirectory(
            self, "select Directory")
        # 비우고 경로 입력
        self.traindatatextEdit.clear()
        self.traindatatextEdit.append(
            '경로: {}'.format(create_sr_data['data_dir']))

        treeModel = QFileSystemModel()
        self.treeView.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.treeView.setRootIndex(treeModel.index(create_sr_data['data_dir']))
        self.treeView.hideColumn(1)
        self.treeView.hideColumn(2)
        self.treeView.hideColumn(3)

    def showImg(self, index):
        self.mainImg = self.treeView.model().filePath(index)
        pixmap = QtGui.QPixmap(self.mainImg).scaled(
            340, 350, Qt.IgnoreAspectRatio)
        self.createSRDataImg.setPixmap(pixmap)

        img = Image.open(self.mainImg)
        self.imgFN.setText(img.filename.split('/')[-1])
        self.imgWH.setText("{} X {}".format(
            str(img.width), str(img.height)))
        self.imgEX.setText(img.format)
        self.imgCN.setText(img.filename.split('/')[-2])
    # model 2

    # def save_dir_save(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.ShowDirsOnly
    #     create_sr_data['save_dir'] = QFileDialog.getExistingDirectory(
    #         self, "select Directory")
    #     # 비우고 경로 입력
    #     self.modeldirtextEdit.clear()
    #     self.modeldirtextEdit.append(
    #         '경로: {}'.format(create_sr_data['save_dir']))

    def goToBack(self):
        widget.setCurrentWidget(choice)


    def batch_changed(self):
        create_sr_data['batch_size'] = self.batchtextEdit.toPlainText()
        if self.batchtextEdit.toPlainText():
            learning['batch_size'] = int(self.batchtextEdit.toPlainText())
            learn_sr.batch_size.setText(
                'Batch Size : {}'.format(self.batchtextEdit.toPlainText()))
        # print(self.batch_size)

    def learning_changed(self):
        create_sr_data['learning_rate'] = self.learningtextEdit.toPlainText()
        try:
            if self.learningtextEdit.toPlainText():
                learning['lr'] = float(self.learningtextEdit.toPlainText())
                # print(self.learningtextEdit.toPlainText())
                learn_sr.learnig_rate.setText('Learning Rate : {}'.format(
                    self.learningtextEdit.toPlainText()))
        except:
            self.warningMSG("주의", "숫자를 입력해주세여.")

    def epoch_changed(self):
        create_sr_data['epoch'] = self.epochtextEdit.toPlainText()
        try:
            if self.epochtextEdit.toPlainText():
                learning['epochs'] = int(self.epochtextEdit.toPlainText()) +1
                learn_sr.epoch.setText('Epoch : {}'.format(
                    self.epochtextEdit.toPlainText()))
        except:
            self.warningMSG("주의", "숫자를 입력해주세여.")

    def model_name_changed(self):
        # print(self.modelnametextEdit.toPlainText())
        create_sr_data['model_name'] = self.modelnametextEdit.toPlainText()
        learning['save'] = self.modelnametextEdit.toPlainText()
        learn_sr.model.setText('Model : {}'.format(
            self.modelnametextEdit.toPlainText()))

    def onRes(self, text):
        create_sr_data['resblock'] = text
        try:
            learning['n_resblocks'] = int(text)
        except:
            self.warningMSG("주의", "숫자를 입력해주세여.")

    def onFeature(self, text):
        create_sr_data['feature_map'] = text
        try:
            learning['n_feats'] = int(text)
        except:
            self.warningMSG("주의", "숫자를 입력해주세여.")

    def onScale(self, text):

        create_sr_data['scale'] = text
        if text == 'x2':
            learning['scale'] = [2]
            learn_sr.rate.setText('배율 : {}'.format(text))

        elif text == 'x3':
            learning['scale'] = [3]
            learn_sr.rate.setText('배율 : {}'.format(text))

        elif text == 'x4':
            learning['scale'] = [4]
            # learning['pre_train'] = './pixby/srtest/experiment/edsr_baseline_x4/model/model_best.pt'
            learn_sr.rate.setText('배율 : {}'.format(text))
        # print(create_sr_data['scale'])

    # createSR 학습하기 버튼
    def dataLoadFn(self):
        if learning['scale'] == [2]:
            if learning['n_resblocks'] == 16:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x2_res_16_feats_32/model/model_best021632.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x2_res_16_feats_64/model/model_best021664.pt'
            
            elif learning['n_resblocks'] == 32:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x2_res_32_feats_32/model/model_best023232.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x2_res_32_feats_64/model/model_best023264.pt'
            
            elif learning['n_resblocks'] == 48:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x2_res_48_feats_32/model/model_best024832.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x2_res_48_feats_64/model/model_best024864.pt'


        elif learning['scale'] == [3]:
            if learning['n_resblocks'] == 16:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x3_res_16_feats_32/model/model_best031632.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x3_res_16_feats_64/model/model_best031664.pt'
            
            elif learning['n_resblocks'] == 32:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x3_res_32_feats_32/model/model_best033232.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x3_res_32_feats_64/model/model_best033264.pt'
            
            elif learning['n_resblocks'] == 48:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x3_res_48_feats_32/model/model_best034832.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x3_res_48_feats_64/model/model_best034864.pt'

        
        
        elif learning['scale'] == [4]:
            if learning['n_resblocks'] == 16:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x4_res_16_feats_32/model/model_best041632.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x4_res_16_feats_64/model/model_best041664.pt'
            
            elif learning['n_resblocks'] == 32:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x4_res_32_feats_32/model/model_best043232.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x4_res_32_feats_64/model/model_best043264.pt'
            
            elif learning['n_resblocks'] == 48:
                if learning['n_feats'] == 32:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x4_res_48_feats_32/model/model_best044832.pt'
                
                elif learning['n_feats'] == 64:
                    learning['pre_train'] = './pixby/srtest/experiment/base_x4_res_48_feats_64/model/model_best044864.pt'


        widget.setCurrentWidget(learn_sr)



class Learn_SR_Model(QMainWindow, learn_ui_form):

    # filename = ''
    # def __init__(self, parent) :
    # super(Learn_SR_Model, self).__init__(parent)
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.gotestbutton.clicked.connect(self.goTest)
        self.golearnbutton.clicked.connect(self.goSR)

        backbutton1 = QPushButton(self)
        backbutton1.move(0, 10)
        backbutton1.resize(80, 80)
        backbutton1.adjustSize()
        backbutton1.setStyleSheet(
            'image:url(img/undo.png);border:0px;background-color:#e7e6e1')
        backbutton1.clicked.connect(self.goToBack)
        
        
        backbutton2 = QPushButton(self)

        backbutton2.move(60, 10)
        backbutton2.resize(150, 150)
        backbutton2.adjustSize()
        backbutton2.setStyleSheet(
            'image:url(img/house.png);border:0px;background-color:#e7e6e1')
        backbutton2.clicked.connect(self.goToHome)




        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.plotLayout.addWidget(self.canvas)
        self.fig_2 = plt.Figure()
        self.canvas_2 = FigureCanvas(self.fig_2)
        self.plotLayout_2.addWidget(self.canvas_2)
        # self.show()
        # x = Thread1(self)
        # x.start()

    # 학습한 SR저장
    # def dataLoadFn(self):
    #     self.textBox_terminal.append(create_sr_data['scale'])
    #     # print(create_sr_data)
    #     # 모델이름 추후수정하기
    #     model_name = create_sr_data['model_name']
    #     scale = create_sr_data['scale']
    #     batch_size = create_sr_data['batch_size']
    #     learning_rate = create_sr_data['learning_rate']
    #     epoch = create_sr_data['epoch']

    #     self.dbName = "db.sqlite3"
    #     self.conn = sqlite3.connect(self.dbName, isolation_level=None)
    #     self.cur = self.conn.cursor()

    #     self.cur.execute("SELECT name from sqlite_master WHERE type='table'")
    #     tables = self.cur.fetchall()
    #     if len(tables) > 0:
    #         print('이미 테이블 있어')
    #     else:
    #         # 모델테이블 생성 임시로test라고 해놓음(모델이름있으면 오류남)
    #         self.cur.execute(
    #             "CREATE TABLE Test(model_name, 배율, batch_size, learning_rate, epoch);")

    #     self.cur.execute(
    #         f"INSERT INTO Test Values('{model_name}', '{scale}', '{batch_size}', '{learning_rate}', '{epoch}');")

    def goToHome(self):
        widget.setCurrentWidget(choice)


    def goToBack(self):
        widget.setCurrentWidget(create_sr)
    #     self.pushButton.clicked.connect(self.dataLoadFn)

    # def dataLoadFn(self):
    #     filename = QFileDialog.getOpenFileName(self, 'Open File', './')
    #     print(filename)
    #     if filename:
    #         self.label_34.setPixmap(QtGui.QPixmap("filename"))
    #         self.label_34.setGeometry(QtCore.QRect(100, 100, width_size, height_size))

    def goSR(self):
        self.textBox_terminal.append('데이터 전처리 시작')

        files = glob.glob(create_sr_data['data_dir'] + '/*')
        f_nums = 1
        save_area = './SRimages/TESTDATA'
        _HR = './SRimages/TESTDATA/TESTDATA_train_HR'
        _LR = './SRimages/TESTDATA/TESTDATA_train_LR_bicubic'
        if os.path.isdir(_HR):
            shutil.rmtree(_HR)
        if os.path.isdir(_LR):
            shutil.rmtree(_LR)
        os.makedirs('./SRimages/TESTDATA/TESTDATA_train_HR', exist_ok=True)
        os.makedirs(
            './SRimages/TESTDATA/TESTDATA_train_LR_bicubic', exist_ok=True)

        for f in files:
            if learning['scale'] == [2]:
                try:
                    img = Image.open(f)

                    if (int(img.width / 2) > 99) and (int(img.height / 2) > 99):
                        os.makedirs(
                            './SRimages/TESTDATA/TESTDATA_train_LR_bicubic/X2', exist_ok=True)
                        title, ext = os.path.splitext(f)
                        img.save(save_area + '/TESTDATA_train_HR/' +
                                 '{0:04d}'.format(f_nums) + ext)
                        img_resize = img.resize(
                            (int(img.width / 2), int(img.height / 2)))
                        img_resize.save(
                            save_area + '/TESTDATA_train_LR_bicubic/X2/' + '{0:04d}'.format(f_nums) + 'x2' + ext)
                        f_nums += 1
                except OSError as e:
                    pass

            elif learning['scale'] == [3]:
                try:
                    img = Image.open(f)
                    if (int(img.width / 3) > 99) and (int(img.height / 3) > 99):
                        os.makedirs(
                            './SRimages/TESTDATA/TESTDATA_train_LR_bicubic/X3', exist_ok=True)
                        title, ext = os.path.splitext(f)
                        img.save(save_area + '/TESTDATA_train_HR/' +
                                 '{0:04d}'.format(f_nums) + ext)
                        img_resize = img.resize(
                            (int(img.width / 3), int(img.height / 3)))
                        img_resize.save(
                            save_area + '/TESTDATA_train_LR_bicubic/X3/' + '{0:04d}'.format(f_nums) + 'x3' + ext)
                        f_nums += 1
                except OSError as e:
                    pass

            elif learning['scale'] == [4]:
                try:
                    img = Image.open(f)
                    if (int(img.width / 4) > 99) and (int(img.height / 4) > 99):
                        os.makedirs(
                            './SRimages/TESTDATA/TESTDATA_train_LR_bicubic/X4', exist_ok=True)
                        title, ext = os.path.splitext(f)
                        img.save(save_area + '/TESTDATA_train_HR/' +
                                 '{0:04d}'.format(f_nums) + ext)
                        img_resize = img.resize(
                            (int(img.width / 4), int(img.height / 4)))
                        img_resize.save(
                            save_area + '/TESTDATA_train_LR_bicubic/X4/' + '{0:04d}'.format(f_nums) + 'x4' + ext)
                        f_nums += 1
                except OSError as e:
                    pass

        f_nums -= 1
        _number = f_nums * 4 // 5
        learning['data_range'] = '1-{}/{}-{}'.format(_number, _number+1, f_nums)
        # f_nums
        self.textBox_terminal.append('전체 데이터 갯수는 {} 입니다'.format(f_nums))
        # self.m_name = create_sr_data['model_name']
        # widget.setCurrentWidget(widget.currentIndex()+1)
        x = Thread1(self)
        x.start()

        self.golearnbutton.setEnabled(False)

    def goTest(self):
        widget.setCurrentWidget(result_sr)


res_ui = resource_path('pixby/ui/res.ui')
res_form = uic.loadUiType(res_ui)[0]


class Result_Model(QMainWindow, res_form):
    def __init__(self, parent):
        super(Result_Model, self).__init__(parent)
        self.setupUi(self)  # for_class2 ui 셋
        self.datas = [ parent.model_1,  parent.model_2, parent.working_path1, parent. working_path2]
        t = Thread4(self)
        t.start()
        # UI
        # self.res1_loss, self.res1_accuracy, self.res1_matrix = parent.res1
        # self.res2_loss, self.res2_accuracy, self.res2_matrix = parent.res2
        # self.res1_loss, self.res1_accuracy = round(
        #     self.res1_loss, 4), round(self.res1_accuracy, 2)
        # self.res2_loss, self.res2_accuracy = round(
        #     self.res2_loss, 4), round(self.res2_accuracy, 2)
        # # 모델 경로 출력
        # uic.loadUi(form_class2,self)
        # 테이블 모델 이름
        # self.name1.setText("모델이름 :" + parent.model_1.split('/')[-1])
        # self.name2.setText("모델이름 :" + parent.model_2.split('/')[-1])

        # self.setTableWidgetData()  # acc, loss
        # 새창 크기 픽스
        self.setFixedWidth(1000)
        self.setFixedHeight(1000)
        self.show()

    def setTableWidgetData(self):
        self.compare_table.setItem(
            0, 0, QTableWidgetItem(str(self.res1_accuracy)))
        self.compare_table.setItem(0, 1, QTableWidgetItem(str(self.res1_loss)))
        self.compare_table.setItem(
            1, 0, QTableWidgetItem(str(self.res2_accuracy)))
        self.compare_table.setItem(1, 1, QTableWidgetItem(str(self.res2_loss)))
        length = len(self.res1_matrix)
        self.model1.setRowCount(length)
        self.model2.setRowCount(length)
        self.model1.setColumnCount(length)
        self.model2.setColumnCount(length)

        for i in range(length):
            for j in range(length):
                self.model1.setItem(i, j, QTableWidgetItem(
                    str(self.res1_matrix[i][j])))

        for i in range(length):
            for j in range(length):
                self.model2.setItem(i, j, QTableWidgetItem(
                    str(self.res2_matrix[i][j])))


class Result_SR_Model(QMainWindow):

    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

    def __init__(self):
        super(Result_SR_Model, self).__init__()
        loadUi('./pixby/ui/resSR.ui', self)
        #  뒤로가기버튼
        backbutton = QPushButton(self)
        backbutton.move(0, 10)
        backbutton.resize(80, 80)
        backbutton.adjustSize()
        backbutton.setStyleSheet(
            'image:url(img/undo.png);border:0px;background-color:#e7e6e1')
        backbutton.clicked.connect(self.goToBack)
        

        speaker = QLabel(self)
        speaker.move(10,890)
        speaker.resize(30,30)
        speaker.setStyleSheet(
            'image:url(img/megaphone.png);border:0px;background-color:#f7f6e7')


        # 이미지가져오기버튼
        self.setImgBtn.clicked.connect(self.setImg)

        # 테이블 행 사이즈 맞추기
        self.testSRTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # testSR버튼
        self.testSRBtn.clicked.connect(self.testSR)

        self.srmodelchangeBtn.clicked.connect(self.changeSR)

        self.treeView.doubleClicked.connect(self.showImg)
        self.treeView_2.doubleClicked.connect(self.showResImg)

        self.pushButton.clicked.connect(self.setResImg)

    
    def goToHome(self):
        widget.setCurrentWidget(choice)
    def warningMSG(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.send_valve_popup_signal.emit(True)

    def changeSR(self):
        self.testSRTable.setRowCount(1)
        model_dir = QFileDialog.getExistingDirectory(self, "select Directory")
        model_name = os.path.basename(model_dir)
        try:
            # test['pre_train'] = model_dir + '/model/model_best.pt'
            path = model_dir + '/model'
            file_list = os.listdir(path)
            file_list_py = [
                file for file in file_list if file.startswith("model_best")]

            s = os.path.split(file_list_py[0])[1]
            scale = s[10:12]
            resblock = s[12:14]
            features = s[14:16]
            # print(scale, resblock, features)

            testing['pre_train'] = model_dir + '/model/' + s
            testing['scale'] = [int(scale)]
            testing['n_resblocks'] = int(resblock)
            testing['n_feats'] = int(features)

            self.testSRTable.setItem(0, 0, QTableWidgetItem(model_name))
            self.testSRTable.setItem(0, 1, QTableWidgetItem('x' + scale[-1]))
            self.testSRTable.setItem(0, 2, QTableWidgetItem(resblock))
            self.testSRTable.setItem(0, 3, QTableWidgetItem(features))

            # else:
            #     self.warningMSG("주의", "정확한 경로를 확인해주세요.")

            # 분류모델 경로
            # s = os.path.split(model_dir + '/model/')

            # title, ext = os.path.splitext(s[1])
            # print(ext)
            # if ext == '.pt':
            #     print('저장')
            # else:
            #     print('왜 안되지?')

        except:
            self.warningMSG("주의", "모델 파일이 아닙니다.")

    def goToBack(self):
        widget.setCurrentWidget(choice)

    def setImg(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        img_dir = QFileDialog.getExistingDirectory(self)
        testing['dir_demo'] = img_dir

        treeModel = QFileSystemModel()
        self.treeView.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.treeView.setRootIndex(
            treeModel.index(img_dir))
        self.treeView.hideColumn(1)
        self.treeView.hideColumn(2)
        self.treeView.hideColumn(3)

    def showImg(self, index):
        self.mainImg = self.treeView.model().filePath(index)
        pixmap = QtGui.QPixmap(self.mainImg).scaled(
            300, 300, Qt.IgnoreAspectRatio)
        self.testSRImg.setPixmap(pixmap)

    def setResImg(self):
        set_dir = './SRimages/CHANGEDDATA/SRresults/results-Demo'
        treeModel2 = QFileSystemModel()
        self.treeView_2.setModel(treeModel2)
        treeModel2.setRootPath(QDir.rootPath())
        self.treeView_2.setRootIndex(treeModel2.index(set_dir))
        self.treeView_2.hideColumn(1)
        self.treeView_2.hideColumn(2)
        self.treeView_2.hideColumn(3)

    def showResImg(self, index):
        self.mainImg = self.treeView_2.model().filePath(index)
        pixmap = QtGui.QPixmap(self.mainImg).scaled(
            300, 300, Qt.IgnoreAspectRatio)
        self.resSRImg.setPixmap(pixmap)

    def testSR(self):
        _data_dir = './SRimages/CHANGEDDATA/SRresults/results-Demo'
        if os.path.isdir(_data_dir):
            shutil.rmtree(_data_dir)

        x = Thread2(self)
        x.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = QStackedWidget()
    windowclass = WindowClass()
    choice = Choice()
    # select_sr_model = Select_SR_Model()
    create_sr = Create_SR_Model()
    learn_sr = Learn_SR_Model()
    result_sr = Result_SR_Model()
    compare_model = Compare_Model()
    create_cnn = Create_CNN_Model()
    train_cnn = Train_CNN()
    widget.addWidget(windowclass)
    widget.addWidget(choice)
    # widget.addWidget(select_sr_model)
    widget.addWidget(create_sr)
    widget.addWidget(learn_sr)
    widget.addWidget(result_sr)
    widget.addWidget(compare_model)
    widget.addWidget(create_cnn)
    widget.addWidget(train_cnn)
    widget.setFixedHeight(960)
    widget.setFixedWidth(1280)
    widget.show()
    app.exec_()
