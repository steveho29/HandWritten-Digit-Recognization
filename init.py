import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch, torchvision
from torch import nn
from torchvision import transforms
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms