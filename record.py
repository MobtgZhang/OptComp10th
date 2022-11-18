import random
import time
import uuid
import base64
import cv2
from PyQt5.QtWidgets import QWidget,QGraphicsScene,QMessageBox,QDesktopWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from facenet.utils import get_encode
from disp_utils import draw_from_database

class RecordMain(QWidget):
    def __init__(self,window_title,img_face):
        super(RecordMain, self).__init__()
        self.icon_path = "./icon.png"
        self.window_title = window_title
        self.img_face = img_face
        self.ui = loadUi("./record.ui",self)
        self.setWindowTitle(self.window_title)
        self.LCDNum = self.ui.LCDNum
        self.LCDNumTimer = QTimer()
        self.LCDNumTimer.timeout.connect(self.lcd_display)
        self.LCDNumTimer.start(1000)

        self.cap_timier = QTimer()
        self.cap_timier.timeout.connect(self.show_viedo)
        self.cap_timier.start()

        self.FaceView = self.ui.FaceView
        self.TempertureBrowser = self.ui.TempertureBrowser
        self.CovidBrower = self.ui.CovidBrower
        self.NameEdit = self.ui.NameEdit
        self.ClassNumberEdit = self.ui.ClassNumberEdit

        self.saveButton = self.ui.SaveButton
        self.QuitButton = self.ui.QuitButton
        self.QuitButton.clicked.connect(self.quit)
        self.saveButton.clicked.connect(self.save_button)

        self.FaceView = self.ui.FaceView
        self.CameraView = self.ui.CameraView
        if img_face is not None:
            qt_img_face = QImage(self.img_face.data.tobytes(),
                                 self.img_face.shape[1],
                                 self.img_face.shape[0],
                                 self.img_face.shape[1] * 3,
                                 QImage.Format_RGB888)
            qt_img_face_out = QPixmap(qt_img_face).scaled(self.FaceView.width(), self.FaceView.height())
            real_scene = QGraphicsScene()
            real_scene.addPixmap(qt_img_face_out)
        else:
            real_scene = None
        self.FaceView.setScene(real_scene)
        self.center()
    def get_options(self,name_of_pic,classnumber,timer,lcd_timer,face_detector,face_encoder,info_database_conn,cap_video,
                  recognition_t,confidence_t,required_size,temperature_address,MLX90614_TOBJ_address,bus_id_num,bus):
        self.parent_timer = timer
        self.parent_lcd_timer = lcd_timer
        self.face_detector = face_detector
        self.face_encoder = face_encoder
        self.info_database_conn = info_database_conn
        self.recognition_t = recognition_t
        self.confidence_t = confidence_t
        self.required_size = required_size
        self.cap_video = cap_video
        self.NameEdit.setText(name_of_pic)
        self.ClassNumberEdit.setText(classnumber)
        self.temperature_address = temperature_address
        self.MLX90614_TOBJ_address = MLX90614_TOBJ_address
        self.bus_id_num = bus_id_num
        self.bus = bus
    def lcd_display(self):
        self.time_now = time.strftime("%Y %m %d %H:%M:%S")
        self.LCDNum.setDigitCount(17)
        self.LCDNum.display(self.time_now)
    def quit(self):
        self.parent_lcd_timer.start(1000)
        self.parent_timer.start(50)
        self.LCDNumTimer.stop()
        self.cap_timier.stop()
        self.close()

    def center(self):
        # 获取屏幕的尺寸信息
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口的尺寸信息
        size = self.geometry()
        # 将窗口移动到指定位置
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)
    def lcd_display(self):
        self.time_now = time.strftime("%Y %m %d %H:%M:%S")
        self.LCDNum.setDigitCount(17)
        self.LCDNum.display(self.time_now)
    def closeEvent(self, event):
        self.quit()
    def show_cv_img(self, frame):
        img_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image_raw = img_frame.copy()
        img_frame_taged,ret_infos = draw_from_database(img_frame.copy(),self.face_detector,self.face_encoder,self.info_database_conn,
                                       self.recognition_t,self.confidence_t,self.required_size)

        # read the temperature
        try:
            tmp_data = self.bus.read_word_data(self.temperature_address, self.MLX90614_TOBJ_address)
            temperature = (tmp_data*0.02) - 273.15
        except IOError as e:
            temperature = -1
        self.TempertureBrowser.setText("%0.2f" % (temperature))
        self.CovidBrower.setText("Yes") # ("Yes" if random.random() > 0.5 else "No")
        if len(ret_infos) > 0:
            self.img_face = ret_infos[0][2]
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
        else:
            # Detect faces
            bboxes = self.face_detector.detect_face(img_frame)
            if len(bboxes) > 0:
                x1, y1, x2, y2 = bboxes[0]['box']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                self.img_face = img_frame[y1:y2, x1:x2]
                pt_1 = (x1, y1)
                pt_2 = (x2, y2)
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
            else:
                self.img_face = None
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
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Failed to capture the video.")
            msgBox.setWindowTitle(self.window_title)
            msgBox.setWindowIcon(QIcon(self.icon_path))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.show()
            msgBox.exec_()
    def save_button(self):
        name_of_pic = self.NameEdit.toPlainText().strip()
        classnumber = self.ClassNumberEdit.toPlainText().strip()
        if name_of_pic == "" or classnumber == "":
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Name or ClassNumber are Empty!")
            msgBox.setWindowTitle(self.window_title)
            msgBox.setWindowIcon(QIcon(self.icon_path))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.show()
            msgBox.exec_()
        else:
            name_ids = str(uuid.uuid4()).upper()
            time_now = self.time_now
            temperature = self.TempertureBrowser.toPlainText()
            covid19result = self.CovidBrower.toPlainText()
            img_raw = self.image_raw
            img_face = self.img_face.copy()
            img_encode = get_encode(self.face_encoder, img_face, self.required_size)
            sql_insert = "INSERT INTO facedataset (IDName,\
                                                                    name,\
                                                                    classnumber,\
                                                                    time,\
                                                                    temperature,\
                                                                    covid19result,\
                                                                    image_raw_base64_bytes,\
                                                                    image_raw_shape,\
                                                                    image_face_base64_bytes,\
                                                                    image_face_shape,\
                                                                    image_emb_base64_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?);"
            image_emb_base64_bytes = base64.b64encode(img_encode)
            image_raw_base64_bytes = base64.b64encode(img_raw)
            img_raw_shape = str(img_raw.shape)
            image_face_base64_bytes = base64.b64encode(img_face)
            img_face_shape = str(img_face.shape)
            cursor = self.info_database_conn.cursor()
            cursor.execute(sql_insert,
                           (name_ids, name_of_pic, classnumber, time_now, temperature, covid19result,
                            image_raw_base64_bytes, img_raw_shape, image_face_base64_bytes, img_face_shape,
                            image_emb_base64_bytes))
            self.info_database_conn.commit()
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("Saved into database!")
            msgBox.setWindowTitle(self.window_title)
            msgBox.setWindowIcon(QIcon(self.icon_path))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.show()
            msgBox.exec_()


