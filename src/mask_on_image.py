import os
import cv2
import numpy as np
import tqdm
from constant import *

if not os.path.exists(MASKED_IMAGES_DIR):
    os.mkdir(MASKED_IMAGES_DIR)

mask_names = os.listdir(MASK_DIR)
# Remove hidden files if any
for f in mask_names:
    if f.startswith("."):
        mask_names.remove(f)
        
        
for mask_name in tqdm.tqdm(mask_names):
    img_name = mask_name[:-3]+"jpg" #images dosyasında isim karşılığını almak için uzantısını .jpg alıyoruz
    
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name), 0).astype(np.uint8)
    image = cv2.imread(os.path.join(IMAGE_DIR, img_name)).astype(np.uint8)
    #0-255 aralığına almak için uint8 yapıyoruz.
    
    cpy_image = image.copy() #orjinal image'ın kopyası
    
    image[mask==100, :] = (255, 0, 125) #color=100 olan mask'ların konumlarını image'da renklendiriyoruz
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)
    #orjinal image ile renklendirilmiş image'ı %50 %50 birleştiriyoruz
    
    cv2.imwrite(os.path.join(MASKED_IMAGES_DIR, mask_name), opac_image)