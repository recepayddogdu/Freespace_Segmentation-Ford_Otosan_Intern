#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 01:19:41 2021

@author: recepaydogdu
"""
import os

IMAGE_DIR = "../test_data/images"
MASK_DIR = "../test_data/masks" #Mask'ların kaydedileceği dosya yolu.
JSON_DIR = "../test_data/jsons" #Annotation dosyalarının dosya yolu.
MASKED_IMAGES_DIR = "../test_data/masked_images"

BATCH_SIZE = 4

#Input Dimensions
HEIGHT = 512
WIDTH = 512

output_shape = (HEIGHT, WIDTH)

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS = 2