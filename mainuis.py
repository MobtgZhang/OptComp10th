import os
import random
import sys
import time
import smbus
import sqlite3
from PyQt5.QtWidgets import  QWidget,QDesktopWidget,QGraphicsScene,QMessageBox,QComboBox
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
import cv2
from disp_utils import get_filename,draw_from_database,get_usb_camera_index
from mtcnn import MTCNN
from record import RecordMain
from facenet import InceptionResNetV2
class MainApp(QWidget):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = loadUi('mainwin.ui', self)
        self.window_title = "Covid19 Test"
        self.screenshots_dir = "./screenshots"
        self.pictures_dir = "./pictures"
        self.icon_path = "./icon.png"
        self.weights_dir = os.path.join("./lite-weights")
        self.temperature_address = 0x5a
        self.MLX90614_TOBJ_address=0x07
        self.bus_id_num = 8
        self.bus = smbus.SMBus(bus=self.bus_id_num)
        self.setWindowIcon(QIcon(self.icon_path))
        if not os.path.exists(self.pictures_dir):
            os.mkdir(self.pictures_dir)

        if not os.path.exists(self.screenshots_dir):
            os.mkdir(self.screenshots_dir)
        self.setMouseTracking(True)    #allow mouse tracking(use during manual mode)
        self.recflag = False  # allows to start camera recording
        self.is_connected = False  # boolean defining if arduino is connected
        self.mirrorflag = False
        self.initUI()  # set up UI( see below )

        self.timer = QTimer()
        self.timerLCD = QTimer()
        self.timer.timeout.connect(self.show_viedo)
        self.timerLCD.timeout.connect(self.lcd_display)
        self.timerLCD.start(1000)
        # 用于记录人脸识别的时间
        self.timerFaceRecognize = QTimer()
        mtcnn_weights_dir = os.path.join(self.weights_dir,"mtcnn")
        self.face_path = "./Faces"
        facenet_weights_path = os.path.join(self.weights_dir,"facenet","facenet_keras_weights.lite")
        self.face_encoder = InceptionResNetV2(facenet_weights_path)
        self.recognition_t = 0.5
        self.confidence_t = 0.99
        self.required_size = (160, 160)
        self.face_shape_size = (640, 480)
        self.face_detector = MTCNN(mtcnn_weights_dir)

        self.data_db_file = "./people.db"
        # 加载数据库
        self.info_database_conn = sqlite3.connect(self.data_db_file)
        cursor = self.info_database_conn.cursor()
        # 查询表是否存在数据库当中
        sql_sent = "SELECT COUNT(*) FROM sqlite_master where type ='table' and name ='facedataset'"
        result = cursor.execute(sql_sent)
        self.info_database_conn.commit()
        number = result.fetchone()[0]
        if number == 0:
            # 如果不存在则，创建数据库
            sql_sent = "CREATE TABLE facedataset (IDName TEXT  PRIMARY KEY,\
                                                                      name TEXT,\
                                                                      classnumber TEXT,\
                                                                      time TEXT, \
                                                                      temperature FLOAT,\
                                                                      covid19result TEXT,\
                                                                      image_raw_base64_bytes BLOB,\
                                                                      image_raw_shape TEXT,\
                                                                      image_face_base64_bytes BLOB,\
                                                                      image_face_shape TEXT,\
                                                                      image_emb_base64_bytes  BLOB)"
            cursor.execute(sql_sent)
            self.info_database_conn.commit()
        self.img_face = None
    def disp_msg(self,msg_str):
        msg_str = "[Message]:\t%s"%msg_str
        self.RecordLabel.setText(msg_str)
    def initUI(self):
        self.setWindowTitle(self.window_title)

        self.ConnectButton = self.ui.ConnectButton
        self.PauseButton = self.ui.PauseButton
        self.ScreenShotButton = self.ui.ScreenShotButton
        self.RecordButton = self.ui.RecordButton
        self.QuitButton = self.ui.QuitButton
        self.MirrorButton = self.ui.MirrorButton

        self.RecordLabel = self.ui.RecordLabel
        self.CameraView  =self.ui.CameraView
        self.FaceView = self.ui.FaceView
        self.CameraList = self.ui.CameraList
        self.camera_list_infos = get_usb_camera_index()
        for data_dict in self.camera_list_infos:
            self.CameraList.addItem(data_dict['name'],data_dict['index'])
        self.LCDNum = self.ui.LCDNum
        self.NameBrowser = self.ui.NameBrowser
        self.ClassNumberBrowser = self.ui.ClassNumberBrowser
        self.Covid19Browser = self.ui.Covid19Browser
        self.TemperatureBrowser = self.ui.TemperatureBrowser

        self.QuitButton.clicked.connect(self.quit)  # bind quit button to quit method
        self.PauseButton.clicked.connect(self.video_pause)  # bind pause button to pause method
        self.ConnectButton.clicked.connect(self.video_button)
        self.ScreenShotButton.clicked.connect(self.screen_shot)
        self.MirrorButton.clicked.connect(self.mirror_set)
        self.RecordButton.clicked.connect(self.save_data_details)
        self.RecordButton.setEnabled(False)

        self.center()

    def center(self):
        # 获取屏幕的尺寸信息
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口的尺寸信息
        size = self.geometry()
        # 将窗口移动到指定位置
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def quit(self):
        self.recflag = False
        self.is_connected = False
        if hasattr(self,"cap_video"):
            self.cap_video.release()
        self.timer.stop()
        self.timerLCD.stop()
        self.info_database_conn.close()
        sys.exit()
    def closeEvent(self, event):
        self.quit()  # call quit method when cross pressed
    def mirror_set(self):
        if self.is_connected:
            self.mirrorflag = not self.mirrorflag
    def lcd_display(self):
        self.time_now = time.strftime("%Y %m %d %H:%M:%S")
        self.LCDNum.setDigitCount(17)
        self.LCDNum.display(self.time_now)
    def print_msgbox(self,info_text,info_type):
        assert info_type in ['warning','information']
        msgBox = QMessageBox()
        if info_type == "warning":
            msgBox.setIcon(QMessageBox.Warning)
        else:
            msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(info_text)
        msgBox.setWindowTitle(self.window_title)
        msgBox.setWindowIcon(QIcon(self.icon_path))
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.show()
        msgBox.exec_()
    def save_data_details(self):
        self.record_ui = RecordMain(self.window_title, self.img_face)
        self.timerLCD.stop()
        self.timer.stop()
        name_of_pic = self.NameBrowser.toPlainText().strip()
        classnumber = self.ClassNumberBrowser.toPlainText().strip()
        self.record_ui.get_options(name_of_pic,classnumber,
                                   self.timer,self.timerLCD,self.face_detector,self.face_encoder,self.info_database_conn,self.cap_video,
                                   self.recognition_t,self.confidence_t,self.required_size,
				   self.temperature_address,self.MLX90614_TOBJ_address,self.bus_id_num,self.bus)
        self.record_ui.show()
        self.record_ui = None
    def screen_shot(self):
        if self.is_connected and self.recflag:
            ret, npImg = self.cap_video.read()
            if self.mirrorflag:
                npImg = cv2.flip(npImg, 1, dst=None)
            if ret:
                try:
                    if not os.path.exists(self.screenshots_dir):
                        os.mkdir(self.screenshots_dir)
                    filename = get_filename(self.screenshots_dir)
                    cv2.imwrite(filename,npImg)
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Information)
                    msgBox.setText("Screen Shot saved in %s "%filename)
                    msgBox.setWindowTitle(self.window_title)
                    msgBox.setWindowIcon(QIcon(self.icon_path))
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.show()
                    msgBox.exec_()
                except Exception as e:
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Warning)
                    msgBox.setText("Error for %s" % str(e))
                    msgBox.setWindowTitle(self.window_title)
                    msgBox.setWindowIcon(QIcon(self.icon_path))
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.show()
                    msgBox.exec_()
            else:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText("Error for loading camera !")
                msgBox.setWindowTitle(self.window_title)
                msgBox.setWindowIcon(QIcon(self.icon_path))
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.show()
                msgBox.exec_()
        else:
            pass
    def show_cv_img(self, frame):
        if self.mirrorflag:
            frame = cv2.flip(frame, 1, dst=None)
        img_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image_raw = img_frame.copy()
        # read the temperature
        try:
            tmp_data = self.bus.read_word_data(self.temperature_address, self.MLX90614_TOBJ_address)
            temperature = (tmp_data*0.02) - 273.15
        except IOError as e:
            temperature = -1
        self.TemperatureBrowser.setText("%0.2f" % (temperature))
        self.Covid19Browser.setText("Yes") # ("Yes" if random.random() > 0.5 else "No")
        img_frame_taged,ret_infos = draw_from_database(img_frame.copy(),self.face_detector,self.face_encoder,self.info_database_conn,
                                       self.recognition_t,self.confidence_t,self.required_size)

        if len(ret_infos) > 0:
            ret_name = ret_infos[0][0]
            ret_classnumber = ret_infos[0][1]
            self.img_face = ret_infos[0][2]
            self.NameBrowser.setText(ret_name)
            self.ClassNumberBrowser.setText(ret_classnumber)

            if self.img_face is None:
                real_scene = None
            else:
                qt_img_face = QImage(self.img_face.data.tobytes(),
                                     self.img_face.shape[1],
                                     self.img_face.shape[0],
                                     self.img_face.shape[1] * 3,
                                     QImage.Format_RGB888)
                qt_img_face_out = QPixmap(qt_img_face).scaled(self.FaceView.width(), self.FaceView.height())
                real_scene = QGraphicsScene()
                real_scene.addPixmap(qt_img_face_out)
            self.disp_msg("Has Recognize!")
        else:
            # Detect faces
            bboxes = self.face_detector.detect_face(img_frame)
            if len(bboxes) > 0:
                x1,y1,x2,y2 = bboxes[0]['box']
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                self.img_face = img_frame[y1:y2,x1:x2]
                pt_1 = (x1,y1)
                pt_2 = (x2,y2)
                img_frame_taged = cv2.rectangle(img_frame, pt_1, pt_2, (0, 0, 255), 2)
                ret_name = "unknown"
                img_frame_taged = cv2.putText(img_frame_taged,ret_name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                qt_img_face = QImage(self.img_face.data.tobytes(),
                                     self.img_face.shape[1],
                                     self.img_face.shape[0],
                                     self.img_face.shape[1] * 3,
                                     QImage.Format_RGB888)
                qt_img_face_out = QPixmap(qt_img_face).scaled(self.FaceView.width(), self.FaceView.height())
                real_scene = QGraphicsScene()
                real_scene.addPixmap(qt_img_face_out)
                self.NameBrowser.setText(ret_name)
                self.ClassNumberBrowser.setText(None)
                self.disp_msg("Not has Recognize!")
            else:
                self.img_face = None
                self.NameBrowser.setText(None)
                self.ClassNumberBrowser.setText(None)
                self.disp_msg("Not Detect Face!")
                real_scene = None
        self.FaceView.setScene(real_scene)
        qt_img_frame = QImage(img_frame_taged.data.tobytes(),
                             img_frame_taged.shape[1],
                             img_frame_taged.shape[0],
                             img_frame_taged.shape[1] * 3,
                             QImage.Format_RGB888)
        qt_img_frame_out = QPixmap(qt_img_frame).scaled(
            self.CameraView.width(), self.CameraView.height())
        scene = QGraphicsScene()
        scene.addPixmap(qt_img_frame_out)
        self.CameraView.setScene(scene)
    def show_viedo(self):
        ret, frame = self.cap_video.read()
        if ret:
            self.show_cv_img(frame)
        else:
            self.disp_msg("Failed to capture the video.")
    def video_button(self):
        if not self.is_connected:
            self.is_connected = True  # stop recording
            self.recflag = True
            self.cameraid = 0 #int(self.CameraList.itemData(self.CameraList.currentIndex()))
            self.cap_video = cv2.VideoCapture(self.cameraid)
            self.cap_video.set(3,self.face_shape_size[0])
            self.cap_video.set(4,self.face_shape_size[1])
            self.RecordButton.setEnabled(True)
            self.CameraList.setEnabled(False)
            self.ConnectButton.setText("Unconnect")  # change pause button text
            self.timer.start(50)
        else:
            self.recflag = False
            self.is_connected = False
            self.timer.stop()
            self.cap_video.release()
            self.CameraView.setScene(None)
            self.FaceView.setScene(None)
            self.RecordButton.setEnabled(False)
            self.CameraList.setEnabled(True)
            self.ConnectButton.setText("Connect")
            self.NameBrowser.setText(None)
            self.ClassNumberBrowser.setText(None)
            self.TemperatureBrowser.setText(None)
            self.Covid19Browser.setText(None)
    def video_pause(self):
        if self.is_connected and not self.recflag:
            self.recflag = True
            self.PauseButton.setText("Pause")
            self.timer.start(50)
        elif self.is_connected and self.recflag:
            self.recflag = False
            self.timer.stop()
            self.PauseButton.setText("Resume")
        else:
            pass

