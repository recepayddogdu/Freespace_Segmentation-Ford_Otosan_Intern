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

def image_augmentation(image_list):
    for image in tqdm_notebook(image_list):
        img = Image.open(image)
        color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)
    
        img_aug = color_aug(img)
        new_path = image.split(".")[0] + "-1.jpg"
        new_path = new_path.replace("image", "image_augmentation")
        img_aug = np.array(img_aug)
        cv2.imwrite(new_path, img_aug)
        
        
def mask_augmentation(mask_list):
    for mask in tqdm_notebook(mask_list):
        msk = cv2.imread(mask)
        new_mask = msk
        newm_path = mask.split(".")[0] + "-1.png"
        newm_path = newmpath.replace("masks", "mask_augmentation")
        cv2.imwrite(newm_path, new_mask)