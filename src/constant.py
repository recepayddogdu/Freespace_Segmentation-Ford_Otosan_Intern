#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 01:19:41 2021

@author: recepaydogdu
"""
import os

DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks") #Mask'ların kaydedileceği dosya yolu.
JSON_DIR = os.path.join(DATA_DIR, "jsons") #Annotation dosyalarının dosya yolu.
MASKED_IMAGES_DIR = os.path.join(DATA_DIR, "masked_images")
MODELS_DIR = "../test_models"
PREDICT_DIR = os.path.join(DATA_DIR, "predicts")
    
BATCH_SIZE = 4

#Input Dimensions
HEIGHT = 512
WIDTH = 512

output_shape = (HEIGHT, WIDTH)
input_shape = (HEIGHT, WIDTH)

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS = 2