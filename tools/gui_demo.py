import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 

#from PyQt5.QtCore import QThread, pyqtSignal, QObject, QEventLoop, QTimer, QRect, QSize
#from PyQt5.QtWidgets import QLabel, QMenu, qApp, QAction, QDesktopWidget, QWidget, QMainWindow, QApplication, QPushButton, QLineEdit,QTextEdit, QFileDialog

from PyQt5.QtGui import QIcon
import numpy as np
from tools.test import test_batch_size
from tools.train import train
import argparse

class mian_UI(QMainWindow):
    def __init__(self):
        super(mian_UI,self).__init__()
        self.initUI()

    def initUI(self):
        
        self.resize(600,600)
        self.showstatus()
        self.setWindowTitle('舰船目标检测')
        self.center()
        self.menu()

        self.icon()
        #self.btn1()
        self.label = QLabel(self)
        #self.label.setText("显示图片")
        self.label.setFixedSize(500,500)
        self.label.move(50,50)

        self.show()
      
    def menu(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('file')
        functionMenu = menuBar.addMenu('tools')


        newAct = QAction('new',self)

        exitAct = QAction('&exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('exit application')
        exitAct.triggered.connect(qApp.quit)
        
        impMenu = QMenu('import', self)
        impAct = QAction('import image', self)
        impAct.triggered.connect(self.loadimage)
        impAct.setStatusTip('please import a image')
        impMenu.addAction(impAct)

        fileMenu.addAction(newAct)
        fileMenu.addMenu(impMenu)
        fileMenu.addAction(exitAct)

        obj_det = QMenu('object detection',self)
        train = QAction('train',self)
        train.triggered.connect(self.train_model)
        test = QAction('test',self)
        test.triggered.connect(self.test_model)
        functionMenu.addMenu(obj_det)

        obj_det.addAction(train)
        obj_det.addAction(test)

    def train_model(self):
        self.two = Train_UI()
    
    def test_model(self):
        self.three = Test_UI()
        
    def showstatus(self):
        self.statusBar().showMessage('Ready')
        
    def icon(self):    
        self.setWindowIcon(QIcon('icon.png'))

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def btn(self):
        QToolTip.setFont(QFont('SansSerif',10))
        #self.setToolTip('This is a <b>QWidgetButton</b> widget')
        btn=QPushButton('button',self)
        btn.setToolTip('this  is a <b> QPushButton</b> widget')
        btn.clicked.connect(self.loadimage)
        btn.resize(btn.sizeHint())
        btn.move(50,50)

    def btn1(self):
        btn1 = QPushButton('exit',self)
        btn1.clicked.connect(QCoreApplication.instance().quit)
        btn1.resize(btn1.sizeHint())
        btn1.move(400,350)

    def loadimage(self):
        print("load image")
        fname,_=QFileDialog.getOpenFileName(self,'choose image','D:/qt_test','image files(*.jpg *.png *.tif)')
        print(fname)

        img = QtGui.QPixmap(fname)
        im = img.toImage()
        #self.show_image_cv(fname)
        self.label.setPixmap(img.scaled(self.label.width(), self.label.height()))
        self.label.show()

    def showDialog(self):
        ok=True
        text, ok = QInputDialog.getText(self, 'Input', 
            'Enter your path:')
        if ok:
            self.le.setText(str(text))

    def showDialog_o(self):
        ok=True
        text, ok = QInputDialog.getText(self, 'Output', 
            'Enter your path:')
        if ok:
            self.le_o.setText(str(text))


class RunThread(QThread):
    trigger = pyqtSignal()

    def __init__(self, 
                 in_put=None, 
                 modelfile=None, 
                 out_put=None, 
                 epoch = None, 
                 Train=True, 
                 device_flag=None, 
                 score_th=None, 
                 pre_trained_dir=None,
                 parent=None):

        super(RunThread, self).__init__()
        self.input = in_put
        self.modelfile = modelfile
        self.output = out_put
        self.epoch = epoch
        self.Train = Train
        if pre_trained_dir is None:
            self.pre_trained_dir = "./pre_trained"
        else:
            self.pre_trained_dir = pre_trained_dir
        self.device_flag = device_flag
        self.score_th = score_th
    def __del__(self):
        self.wait()

    def run(self):
        if self.Train:
            train(root_dir=self.input, 
                  model_save_dir=self.output, 
                  epochs_num=self.epoch, pretrained_model_dir=self.pre_trained_dir)
        else:
            test_batch_size(img_file=self.input, 
                            model_file=self.modelfile, 
                            result_dir=self.output, 
                            score_th=self.score_th, pretrained_model_dir=self.pre_trained_dir, flag=self.device_flag)
        self.trigger.emit()

class EmittingStr(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec_()

class Train_UI(QWidget):
    def __init__(self, pre_trained_dir=None):
        super(Train_UI, self).__init__()
        self.train_dir = None
        self.save_dir = None
        self.epoch_num = 12
        self.initUI()
        self.pre_trained_model = pre_trained_dir

    def initUI(self):
        
        self.label_le = QLabel(self)
        self.label_le.setText("choose a train dataset")
        self.le = QLineEdit(self)
        #self.le.setPlaceholderText("include images/ and labelxmls/")
        self.le.textChanged.connect(self.enterPress_t)
        self.btn = QPushButton('...', self)
        self.btn.clicked.connect(self.get_dir)

        self.label_le_o = QLabel(self)
        self.label_le_o.setText("choose a path to save model")
        self.le_o = QLineEdit(self)
        #self.le_o.setPlaceholderText("save model files")
        self.le_o.textChanged.connect(self.enterPress_t_o)
        self.btn_o = QPushButton('...', self)
        self.btn_o.clicked.connect(self.get_dir_o)
        
        # set epochs_num
        self.label_le_e = QLabel(self)
        self.label_le_e.setText("set epoch number")
        self.le_e = QLineEdit(self)
        self.le_e.setPlaceholderText("default epoch=12")
        self.le_e.textChanged.connect(self.enterPress_e)

        self.train = QPushButton('start train', self)
        self.train.clicked.connect(self.train_work)

        self.textBrowser = QTextEdit(self)
        self.textBrowser.setReadOnly(True)
        self.textBrowser.setMaximumSize(QSize(16777215, 16777215))
        
        self.clear_btn = QPushButton('Clear',self)
        self.clear_btn.clicked.connect(self.textBrowser.clear)

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

        #布局设置
        grid = QGridLayout()
        #grid.setSpacing(10)
        grid.addWidget(self.label_le,1,0)
        grid.addWidget(self.label_le_o,2,0)
        grid.addWidget(self.label_le_e,3,0)

        grid.addWidget(self.btn,1,6)
        grid.addWidget(self.btn_o,2,6)
        #grid.addWidget(self.btn_e,3,6)

        grid.addWidget(self.train,3,6)
        grid.addWidget(self.clear_btn,8,6)

        grid.addWidget(self.le,1,1,1,5)
        grid.addWidget(self.le_o,2,1,1,5)
        grid.addWidget(self.le_e,3,1,1,5)

        grid.addWidget(self.textBrowser,5,0,2,7)
        
        self.setLayout(grid)

        self.setGeometry(300, 100, 800, 600)
        self.setWindowTitle('目标检测-训练')
        self.show()

    def enterPress_t(self,text):
        self.train_dir = text
    def enterPress_t_o(self,text):
        self.save_dir = text
    
    def enterPress_e(self,text):
        self.epoch_num = np.int64(text)


    def train_work(self):
        self.thread_train = RunThread(in_put=self.train_dir,
                                      out_put=self.save_dir, 
                                      epoch = self.epoch_num, 
                                      Train=True, 
                                      pre_trained_dir=self.pre_trained_model)
        self.thread_train.start()

    def get_dir(self):
        self.train_dir = QFileDialog.getExistingDirectory(self,'选取文件夹',"./")   
        self.le.setText(str(self.train_dir))
    def get_dir_o(self):
        self.save_dir = QFileDialog.getExistingDirectory(self,'选取文件夹',"./")   
        self.le_o.setText(str(self.save_dir))

    def outputWritten(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def printf(self,mypstr):
        self.textBrowser.append(mypstr)
        self.cursot = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursot.End)

class Test_UI(QWidget):
    def __init__(self, pre_trained_dir=None):
        super(Test_UI,self).__init__()
        self.images_dir = None
        self.model_file = None
        self.results_dir = None
        self.score_th = 0.3
        self.pre_trained_model = pre_trained_dir
        self.initUI()
        #self.flag = 'AUTO'
    def initUI(self):

        self.label_le1 = QLabel(self)
        self.label_le1.setText("Input a image file")
        self.le1 = QLineEdit(self)
        #self.le1.setPlaceholderText("choose a image file")
        self.le1.textChanged.connect(self.enterPress_imgs)
        self.btn1 = QPushButton('...', self)
        self.btn1.clicked.connect(self.get_img_dir)

        self.label_le2 = QLabel(self)
        self.label_le2.setText("load a model file")
        self.le2 = QLineEdit(self)
        #self.le2.setPlaceholderText("load a model file (*.pth)")
        self.le2.textChanged.connect(self.enterPress_pth)
        self.btn2 = QPushButton('...', self)
        self.btn2.clicked.connect(self.get_pth)

        self.label_le3 = QLabel(self)
        self.label_le3.setText("choose a path to save results")
        self.le3 = QLineEdit(self)
        #self.le3.setPlaceholderText("choose a path for all(*.xml)")
        self.le3.textChanged.connect(self.enterPress_rls)
        self.btn3 = QPushButton('...', self)
        self.btn3.clicked.connect(self.get_rls_dir)

        self.label_le4 = QLabel(self)
        self.label_le4.setText("set score threshold")
        #self.btn4 = QPushButton('score_th', self)
        self.le4 = QLineEdit(self)
        self.le4.setPlaceholderText("default 0.3")
        self.le4.textChanged.connect(self.enterPress_score_th)

        #命令窗口设置
        self.textBrowser = QTextEdit(self)
        self.textBrowser.setReadOnly(True)
        self.textBrowser.setMaximumSize(QSize(16777215, 16777215))
        self.clear_btn = QPushButton('Clear',self)
        self.clear_btn.clicked.connect(self.textBrowser.clear)

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

        self.label_cb = QLabel(self)
        self.label_cb.setText("set device (GPU/CPU)")
        self.cb = QComboBox(self)
        self.cb.addItem('AUTO')
        self.cb.addItem('ONLY_CPU')
        self.cb.addItem('NO_CUDNN')
        #信号
        self.flag = self.cb.currentText()
        self.cb.currentIndexChanged[str].connect(self.print_value)

        self.test = QPushButton('start test', self)
        self.test.clicked.connect(self.test_work)

        #布局设置
        grid = QGridLayout()
        grid.addWidget(self.label_le1,1,0)
        grid.addWidget(self.label_le2,2,0)
        grid.addWidget(self.label_le3,3,0)
        grid.addWidget(self.label_le4,4,0)
        grid.addWidget(self.label_cb,4,4)

        grid.addWidget(self.btn1,1,6)
        grid.addWidget(self.btn2,2,6)
        grid.addWidget(self.btn3,3,6)

        
        grid.addWidget(self.cb,4,5)
        grid.addWidget(self.test,4,6)
        grid.addWidget(self.clear_btn,8,6)

        grid.addWidget(self.le1,1,1,1,5)
        grid.addWidget(self.le2,2,1,1,5)
        grid.addWidget(self.le3,3,1,1,5)
        grid.addWidget(self.le4,4,1,1,2)
        grid.addWidget(self.textBrowser,5,0,2,7)
        
        self.setLayout(grid)

        self.setGeometry(300, 100, 800, 600)
        self.setWindowTitle('目标检测-检测')
        self.show()

    def get_img_dir(self):
        self.images_dir,_ = QFileDialog.getOpenFileName(self,'选择测试图像', "./:", 'image file(*.jpg; *.png; *.tif; *.tiff);; all files (*)')
        #print(self.images_dir)
        self.le1.setText(str(self.images_dir))
    def get_pth(self):
        self.model_file,_ = QFileDialog.getOpenFileName(self,'选取模型文件', "./:", 'model file(*.pth)')
        self.le2.setText(str(self.model_file))
    def get_rls_dir(self):
        self.results_dir = QFileDialog.getExistingDirectory(self,'选取文件夹',"./")   
        self.le3.setText(str(self.results_dir))

    def enterPress_imgs(self,text):
        self.images_dir = text
    def enterPress_pth(self,text):
        self.model_file = text
    def enterPress_rls(self,text):
        self.results_dir = text
    def enterPress_score_th(self,text):
        self.score_th = np.float(text)

    def test_work(self):
        #print("testing")
        self.thread_test = RunThread(in_put=self.images_dir, 
                                     modelfile=self.model_file, 
                                     out_put=self.results_dir, 
                                     Train=False, 
                                     device_flag=self.flag, 
                                     score_th=self.score_th, 
                                     pre_trained_dir=self.pre_trained_model)
        self.thread_test.start()

    def outputWritten(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def printf(self,mypstr):
        self.textBrowser.append(mypstr)
        self.cursot = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursot.End)

    def print_value(self, value):
        self.flag = value
        #print("device: %s" % self.flag)
        
class two(QWidget):

    def __init__(self):
        super().__init__()    
        self.initUI()
        
        
    def initUI(self):

        self.btn = QPushButton('Dialog', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.showDialog)
        
        self.le = QLineEdit(self)
        self.le.move(130, 22)
        
        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('Input dialog')
        self.show()
        
        
    def showDialog(self):
        
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            'Enter your name:')
        
        if ok:
            self.le.setText(str(text))

def parse_args():
    parser = argparse.ArgumentParser(description='Test or Train')
    parser.add_argument('--operation', help="the dir to save result file")
    parser.add_argument('--pretrained_dir', help="choose path of pretrained model")
    args = parser.parse_args()
    return args

def main():
    #根据输入参数选择训练或者是测试
    operate_set = ['train', 'test']
    args = parse_args()
    if args.operation == operate_set[0]:
        app = QApplication(sys.argv)
        ex=Train_UI(pre_trained_dir=args.pretrained_dir)
        sys.exit(app.exec_())
    if args.operation == operate_set[1]:
        app = QApplication(sys.argv)
        ex=Test_UI(pre_trained_dir=args.pretrained_dir)
        sys.exit(app.exec_())
      

if __name__=="__main__":
    main()
    
#.\objectDetectionGUI.exe --operation test --pretrained_dir d:/algorithm/pre_trained