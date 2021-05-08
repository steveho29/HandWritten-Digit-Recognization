import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch, torchvision
from torch import nn, optim, utils
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import torch, torchvision, numpy
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
import cv2, imutils

