import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import imutils


CONVOLUTIONAL_MODEL = 'convolutional_model.pt'
FULLY_CONNECTED_MODEL = 'fully_connected_model.pt'
IS_CONVO_MODEL = True


def classify_image(img):
    model_path = CONVOLUTIONAL_MODEL
    if not IS_CONVO_MODEL:
        model_path = FULLY_CONNECTED_MODEL
        img = img.view(1, 784)
    model = torch.load(model_path)

    with torch.no_grad():
        logpb = model(img)
    pb = torch.exp(logpb)
    probab = list(pb.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    print("Probability =", max(probab))
    return probab.index(max(probab)), max(probab)


def sort_contours(cnts, reverse=False):
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes),
                                       key=lambda b: b[1][0], reverse=reverse))
    return cnts, bounding_boxes


def detect_digit(filename, idx):
    list_digits = ''
    cv_img = cv2.imread(filename)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return '0'
    cnts, _ = sort_contours(cnts, reverse=False)

    digitCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        # Check size of contour whether it is big enough to be a digit
        if w >= 10 and h >= 50:
            digitCnts.append(c)

    output = cv_img.copy()

    count = 0

    # Detect every single digit
    for cnt in digitCnts:
        count += 1
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h + 10), (255, 0, 0), 3)

        k = int(max(w, h) * 1.7)
        new_width = k
        new_height = k
        mid_x = int((k - w) / 2)
        mid_y = int((k - h) / 2)
        i = 0
        digit = np.full((new_height, new_width), 0.0)

        # Center digit
        for axis_x in range(x, x + w):
            j = 0
            for axis_y in range(y, y + h):
                digit[mid_y + j][mid_x + i] = gray[axis_y][axis_x]
                j += 1
            i += 1
        idx += 1

        # cv2.imshow(('Digit' + str(idx)), output)
        # cv2.imshow(('Crop Digit ' + str(idx)), digit)
        # cv2.imwrite('digit.png', digit)

        img = Image.fromarray(digit).convert('L')
        img = transforms.Resize((28, 28))(img)
        img = transforms.ToTensor()(img)
        # img = F.pad(img, [4, 4, 4, 4], 'constant', 0)
        val = 0.1
        if h >= 350:
            val = 0.07

        # print('Normalize Value: {}'.format(val))
        img = transforms.Normalize((val,), (val,))(img)
        img = img.unsqueeze(0)

        # torchvision.utils.save_image(img, 'digit' + str(idx) + '.png')

        predict_number, prob = classify_image(img)
        list_digits += str(predict_number)

    # cv2.imshow("Output", cv_img)
    print(list_digits)
    if not len(list_digits):
        list_digits = '0'
    return list_digits


class Draw(QWidget):
    def __init__(self, window, is_left):
        super().__init__()
        self.pix = QPixmap()
        self.lastPoint = QPoint(-10, -10)
        self.endPoint = QPoint(-10, -10)
        self.window = window
        self.idx = 0
        self.m_width = 500
        self.m_height = 500
        self.is_left = is_left
        self.isActive = False
        if is_left:
            self.filename = 'left.png'
        else:
            self.filename = 'right.png'

        pen_pix_map = QPixmap('pen.png').scaled(QSize(20, 20))
        cursor = QCursor(pen_pix_map)
        self.setCursor(cursor)

        self.initUI()

    def initUI(self):
        # self.resize(600, 500)
        self.setMinimumSize(self.m_width, self.m_height)
        self.setMaximumSize(self.m_width, self.m_height)
        self.pix = QPixmap(self.m_width, self.m_height)
        self.pix.fill(Qt.black)

    # Drawing
    def paintEvent(self, event):
        p = QPainter(self.pix)
        p.setPen(QPen(Qt.white, 5))
        # Draw a straight line two positions before and after the mouse pointer
        # Minus 10 to make sure the line start at the head of the pen
        self.endPoint.setX(self.endPoint.x()-10)
        self.endPoint.setY(self.endPoint.y()+10)
        p.drawLine(self.lastPoint, self.endPoint)

        # Make the previous value equal to the next value to draw a continuous line
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    # This function is triggered when the left mouse button moves
    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()  # Call paintEvent function, repaint

    # This function is triggered when the left mouse button is released
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()  # Call paintEvent function, repaint
        file = QFile(self.filename)
        file.open(QIODevice.WriteOnly)
        self.pix.save(file, "PNG")
        self.isActive = True
        self.window.calculate()


class Main(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("app.ui", self)
        self.draw_left = Draw(self, True)
        self.draw_right = Draw(self, False)
        self.draw_box_left.addWidget(self.draw_left)
        self.draw_box_right.addWidget(self.draw_right)
        self.clear_button.clicked.connect(self.clear)
        self.left_digits = ''
        self.right_digits = ''
        self.operation = {0: '+', 1: '-', 2: '*', 3: '/'}
        self.enter.clicked.connect(self.calculate)
        self.setMaximumSize(1080,720)
        self.setWindowTitle('Super Calculator')
        self.setWindowIcon(QIcon('logo.ico'))
        self.model_box.setStyleSheet('color: black;\n background-color: pink;')

        # x = QComboBox()
        # x.palette().highlightedText()
        # self.operator_box.palette().setColorGroup(Qt.white)
        # self.operator_box.palette().setColor(Qt.white)
        self.operation_dict = {
            0: lambda x, y: x + y,
            1: lambda x, y: x - y,
            2: lambda x, y: x * y,
            3: lambda x, y: x / y,
        }
        self.model_box.currentIndexChanged.connect(self.setModel)
        self.operator_box.currentIndexChanged.connect(self.calculate)

    def setModel(self):
        global IS_CONVO_MODEL
        if self.model_box.currentIndex() == 0:
            IS_CONVO_MODEL = True
            self.model_box.setStyleSheet('color: black;\n background-color: pink;')

        else:
            IS_CONVO_MODEL = False
            self.model_box.setStyleSheet('color: white;\n background-color: brown;')

        self.calculate()
    def clear(self):
        self.draw_left.pix.fill(Qt.black)
        self.draw_left.lastPoint = QPoint(-10, -10)
        self.draw_left.endPoint = QPoint(-10, -10)
        self.draw_left.update()
        self.draw_left.idx = 0
        self.draw_right.isActive = False

        self.draw_right.pix.fill(Qt.black)
        self.draw_right.lastPoint = QPoint(-10, -10)
        self.draw_right.endPoint = QPoint(-10, -10)
        self.draw_right.update()
        self.draw_right.idx = 0
        self.draw_left.isActive = False

        self.result_box.setText('')
        self.left_digits = self.right_digits = ''

    def calculate(self):
        if not self.draw_left.isActive:
            self.left_digits = '0'
        else:
            self.left_digits = detect_digit('left.png', 0)
        if not self.draw_right.isActive:
            self.right_digits = '0'
        else:
            self.right_digits = detect_digit('right.png', len(self.left_digits))
        result = self.operation_dict[self.operator_box.currentIndex()](int(self.left_digits), int(self.right_digits))
        expression = self.left_digits + \
                     self.operation[self.operator_box.currentIndex()] + \
                     self.right_digits + '=' + str(result)
        print(expression)
        self.result_box.setText(expression)

        self.draw_right.lastPoint = QPoint(-10, -10)
        self.draw_right.endPoint = QPoint(-10, -10)

        self.draw_left.lastPoint = QPoint(-10, -10)
        self.draw_left.endPoint = QPoint(-10, -10)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Main()
    form.show()
    sys.exit(app.exec_())
