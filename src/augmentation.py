import numpy as np
import cv2
import json
import os
import torch
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage import transform
import random 
from torchvision import transforms as T
from PIL import Image
from constant import *
from train import *

def image_augmentation(image_list, color_aug=True, flipUD=True, rotate=True, ):
  