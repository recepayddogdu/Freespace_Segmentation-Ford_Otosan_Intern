import glob
import cv2
import torch
import numpy as np
from constant import *

def tensorize_image(image_path_list, output_shape, cuda=False):
    local_images = [] # Create empty list
    
    for image_path in image_path_list: # For each image
    
        img = cv2.imread(image_path) # Access and read image
        
        img = cv2.resize(img, output_shape) # Resize the image according to defined shape
        
        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)
        
        local_images.append(torchlike_image) # Add into the list
    
    # Convert from list structure to torch tensor
    image_array = np.array(local_images, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()
    
    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()
        
    return torch_image

def tensorize_mask(mask_path_list, output_shape, N_CLASS, cuda=False):
    local_masks = []
    
    for mask_path in mask_path_list:
        
        # Access and read mask
        mask = cv2.imread(mask_path, 0)
        
        # Resize the image according to defined shape
        mask = cv2.resize(mask, output_shape)
        
        #Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, N_CLASS)
        
        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)
        
        local_masks.append(torchlike_mask)
    
    mask_array = np.array(local_masks, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    
    if cuda:
        torch_mask = torch_mask.cuda()
    
    return torch_mask



































