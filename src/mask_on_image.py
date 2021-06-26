import os
import cv2
import numpy as np
import tqdm

MASK_DIR = "../data/masks"

MASKED_IMAGES_DIR = "../data/masked_images"
if not os.path.exists(MASKED_IMAGES_DIR):
    os.mkdir(MASKED_IMAGES_DIR)

IMAGE_DIR = "../data/images"

mask_names = os.listdir(MASK_DIR)
# Remove hidden files if any
for f in mask_names:
    if f.startswith("."):
        mask_names.remove(f)
        
        
for mask_name in tqdm.tqdm(mask_names):
    img_name = mask_name[:-3]+"jpg" 
    
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name), 0).astype(np.uint8)
    image = cv2.imread(os.path.join(IMAGE_DIR, img_name)).astype(np.uint8)
    
    
    cpy_image = image.copy()
    
    image[mask==100, :] = (255, 0, 125)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)
    
    cv2.imwrite(os.path.join(MASKED_IMAGES_DIR, mask_name), opac_image)