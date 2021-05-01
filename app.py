import sys
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtWidgets import*
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch, torchvision, numpy
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
# from Trainning import *

model_path = 'convolutional_model.pt'


class MnistModel(nn.Module):
    """
    Custom CNN Model for Mnist
    """

    def __init__(self, classes: int) -> None:
        super(MnistModel, self).__init__()

        self.classes = classes

        # initialize the layers in the first (CONV => RELU) * 2 => POOL + DROP
        # (N,1,28,28) -> (N,16,24,24)
        self.conv1A = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        # (N,16,24,24) -> (N,32,20,20)
        self.conv1B = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        # (N,32,20,20) -> (N,32,10,10)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU()
        self.do = nn.Dropout(0.25)

        # initialize the layers in the second (CONV => RELU) * 2 => POOL + DROP
        # (N,32,10,10) -> (N,64,8,8)
        self.conv2A = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        # (N,64,8,8) -> (N,128,6,6)
        self.conv2B = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        # (N,128,6,6) -> (N,128,3,3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # initialize the layers in our fully-connected layer set
        # (N,128,3,3) -> (N,32)
        self.dense3 = nn.Linear(128*3*3, 32)

        # initialize the layers in the softmax classifier layer set
        # (N, classes)
        self.dense4 = nn.Linear(32, self.classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(x)
        x = self.act(x)
        x = self.conv1B(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.do(x)

        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv2A(x)
        x = self.act(x)
        x = self.conv2B(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.do(x)

        # build our FC layer set
        x = x.view(x.size(0), -1)
        x = self.dense3(x)
        x = self.act(x)
        x = self.do(x)

        # build the softmax classifier
        x = nn.functional.log_softmax(self.dense4(x), dim=1)
        return x


def test_and_show_image(model_path, img):
    model = torch.load(model_path)
    # img = img.view(1, 784)
    with torch.no_grad():
        logpb = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    pb = torch.exp(logpb)
    probab = list(pb.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    # classify(img.view(1, 28, 28), pb)
    return probab.index(max(probab)), max(probab)


def test_and_show_image_linear(model_path, img):
    model = torch.load(model_path)
    img = img.view(1, 784)
    with torch.no_grad():
        logpb = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    pb = torch.exp(logpb)
    probab = list(pb.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    # classify(img.view(1, 28, 28), pb)
    return probab.index(max(probab)), max(probab)


class Draw(QWidget):
    def __init__(self, window):
        super().__init__()
        self.setWindowTitle('Simple drawing')
        self.pix = QPixmap()
        self.lastPoint = QPoint(-10, -10)
        self.endPoint = QPoint(-10, -10)
        self.window = window
        self.initUI()

    def initUI(self):
        # self.resize(600, 500)
        self.setMinimumSize(280, 280)
        self.setMaximumSize(280, 280)
        self.pix = QPixmap(280, 280)
        self.pix.fill(Qt.black)

    # Drawing
    def paintEvent(self, event):
        p = QPainter(self.pix)
        p.setPen(QPen(Qt.white, 10))
        p.drawLine(self.lastPoint, self.endPoint)  # Draw a straight line according to the two positions before and after the mouse pointer
        self.lastPoint = self.endPoint  # Make the previous coordinate value equal to the next coordinate value to draw a continuous line
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    #This function is triggered when the left mouse button moves
    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()#Call paintEvent function, repaint

    #This function is triggered when the left mouse button is released
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()  # Call paintEvent function, repaint
        file = QFile('image.png')
        file.open(QIODevice.WriteOnly)
        self.pix.save(file, "PNG")
        p = transforms.Compose([transforms.Resize((28, 28)), transforms.Normalize((0.5,), (0.5,))])

        image_fp = open('image.png', "rb")
        img = Image.open(image_fp).convert('L')
        # img.convert('L')
        img = transforms.ToTensor()(img)
        img = p(img)
        # img = img.unsqueeze(0)
        # img = torchvision.transforms.F.pad(img, (4,4,4,4), 0, 'constant')
        print(img.shape)
        # transforms.Normalize(img, (0.5,), (0.5,))
        # print(img)
        torchvision.utils.save_image(img, 'haha.png')

        # print(model.forward(img))
        predict_number, prob = test_and_show_image_linear('model.pt', img)
        self.window.label.setText(str(predict_number))
        self.window.lineEdit.setText(str(prob))

class Main(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("main.ui", self)
        self.draw = Draw(self)
        self.verticalLayout_6.addWidget(self.draw)
        self.clear_button.clicked.connect(self.clear)

    def clear(self):
        self.draw.pix.fill(Qt.black)
        self.draw.lastPoint = QPoint(-10, -10)
        self.draw.endPoint = QPoint(-10, -10)
        self.draw.update()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Main()
    form.show()
    sys.exit(app.exec_())
