import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QPushButton, QLabel, QVBoxLayout, QFileDialog
import cv2
import threading
from PIL import Image
import time
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from helpers import mosaic_blur_multiple
# from mtcnn_model import extract_faces, draw_rectangles
from yolov5 import detect
from facenet_pytorch import MTCNN
from mtcnn_detect import extract_faces

model_face = MTCNN(keep_all=True, post_process=True).eval()


def to_bytes(non_bytes_img, type='jpeg'):
    from io import BytesIO
    buf = BytesIO() 
    non_bytes_img.save(non_bytes_img, type)
    buf.seek(0)
    image_bytes = buf.read()
    buf.close()
    return image_bytes

def swapxy(faces, shape, toint=True):
    for i, f in enumerate(faces):
        # faces[i] = [max(f[1], 0), max(f[0], 0), min(f[3], shape[0]), min(f[2], shape[1])]
        faces[i] = [max(f[0], 0), max(f[1], 0), min(f[2], shape[0]), min(f[3], shape[1])]
        if toint:
            faces[i] = [int(o) for o in faces[i]]
    return faces


class QtCapture(QWidget):
    def __init__(self, *args):
        super(QWidget, self).__init__()

        self.fps = 24
        self.cap = cv2.VideoCapture(*args)

        self.video_frame = QLabel()
        lay = QVBoxLayout()
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

        
        self.isCapturing = False
        self.ith_frame = 1
        

    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        ret, frame = self.cap.read()

        # Save images if isCapturing
        if self.isCapturing:
            cv2.imwrite('img_%05d.jpg'%self.ith_frame, frame)
            self.ith_frame += 1
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        face_detections = model_face.detect(frame)
        if face_detections[0] is None:
            fixed_detections_xyxy = []
        else:
            fixed_detections_xyxy = swapxy(list(face_detections[0]), shape=frame.shape[:2])

        # face_detections = extract_faces(frame)

        # Advertising detection
        # ad_detections = model_ad.detect(frame)
        ad_detections = detect.detect_custom(frame, weights='yolov5/weights/best.pt').tolist()
        trimed_ad_detections = [det[:4] for det in ad_detections]
        # ad_detections = []
        # print('ad', ad_detections)
        # print('face', fixed_detections_xyxy+trimed_ad_detections)
        detections = fixed_detections_xyxy + trimed_ad_detections
        print(detections)

        self.blur = False
        # Draw detections on image
        if self.blur:
            frame = mosaic_blur_multiple(frame, detections)
        else:
            print(detections)
            for x1, y1, x2, y2 in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255))

        # t1 = threading.Thread(target=QtGui.QImage, args=(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888))
        # t1.start()
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)
        self.count += 1
        print((time.time()-self.t1)/self.count)
        

    # def streamThread(self, frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888):
    #     while 

    def start(self):
        self.t1 = time.time()
        self.count = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    # ------ Modification ------ #
    def capture(self):
        if not self.isCapturing:
            self.isCapturing = True
        else:
            self.isCapturing = False
    # ------ Modification ------ #

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()

    

class ControlWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.capture = None

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.picture_button = QPushButton('Take picture')
        self.picture_button.clicked.connect(self.picture)
        self.quit_button = QPushButton('End')
        self.quit_button.clicked.connect(self.endCapture)
        self.end_button = QPushButton('Stop')

        self.inference_file_button = QPushButton('Inference file')
        self.inference_file_button.clicked.connect(self.inf_file)
        # self.dialog = QFileDialog()
        self.le = QLabel('')
        
        # ------ Modification ------ #
        self.capture_button = QPushButton('Capture')
        self.capture_button.clicked.connect(self.saveCapture)
        # ------ Modification ------ #

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.picture_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.inference_file_button)
        vbox.addWidget(self.le)
        # vbox.addWidget(self.dialog)
        vbox.addWidget(self.quit_button)

        # ------ Modification ------ #
        vbox.addWidget(self.capture_button)
        # ------ Modification ------ #

        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(0)
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    # ------ Modification ------ #
    def saveCapture(self):
        if self.capture:
            self.capture.capture()
    # ------ Modification ------ #
    def inf_file(self):
        filename = QFileDialog.getOpenFileName()[0]
        self.le.setPixmap(QtGui.QPixmap(filename))

        im  = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        faces = extract_faces(im)
        
        im_mod = draw_rectangles(im, faces, enlarge=True)
        cv2.imwrite('im_mod.jpg', im_mod)
        # except:
        #     print('wrong file')


    def picture(self):
        if not self.capture:
            self.capture = QtCapture(0)
            self.end_button.clicked.connect(self.capture.stop)
        ret, frame = self.capture.cap.read()

        cv2.imwrite('img.jpg', frame)

        # My webcam yields frames in BGR format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Face detection
        frame = draw_rectangles(frame, extract_faces(frame), enlarge=True)
        cv2.imwrite('img_mod.jpg', frame)


if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())