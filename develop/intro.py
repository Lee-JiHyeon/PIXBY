import sys
# from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5 import uic
from pixby.newSR import Create_SR_Model
from pixby.compare import compareModel, resultModel
from pixby.selectSR import Select_SR_Model
# form_class = uic.loadUiType("intro.ui")[0]


class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('./pixby/ui/intro.ui', self)

        self.startBtn.clicked.connect(self.gotoChoice)
        self.compare.clicked.connect(self.gotoCompare)

    def gotoChoice(self):
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotoCompare(self):
        widget.setCurrentIndex(widget.currentIndex()+4)


class Choice(QMainWindow):
    def __init__(self):
        super(Choice, self).__init__()
        loadUi('./pixby/ui/choice.ui', self)

        self.gotoModelBtn.clicked.connect(self.gotoModel)
        self.goToCreateSR.clicked.connect(self.gotoCreateSR)

    def gotoModel(self):
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotoCreateSR(self):
        widget.setCurrentIndex(widget.currentIndex()+2)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = QStackedWidget()
    windowclass = WindowClass()
    choice = Choice()
    select_sr_model = Select_SR_Model()
    create_sr = Create_SR_Model()
    compare_model = compareModel()
    result_model = resultModel()
    widget.addWidget(windowclass)
    widget.addWidget(choice)
    widget.addWidget(select_sr_model)
    widget.addWidget(create_sr)
    widget.addWidget(compare_model)
    widget.addWidget(result_model)
    widget.setFixedHeight(960)
    widget.setFixedWidth(1280)
    widget.show()
    app.exec_()
