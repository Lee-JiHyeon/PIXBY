from os import pardir
from sqlite3.dbapi2 import connect
import sys
from PyQt5 import uic
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *

# db연결
import sqlite3

from PyQt5.uic.uiparser import QtWidgets


class Select_SR_Model(QMainWindow):
    def __init__(self):
        super(Select_SR_Model, self).__init__()
        loadUi('./pixby/ui/select.ui', self)
        # sql 연동
        self.sqlConnect()

        # 이미지 열기 버튼
        self.imageBtn.clicked.connect(self.openImage)

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
