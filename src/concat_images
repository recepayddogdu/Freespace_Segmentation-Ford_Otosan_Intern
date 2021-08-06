import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

image_list_path = "../data/for_gif/images"
mask_list_path = "../data/for_gif/masks"
predict_list_path = "../data/for_gif/predicts"
concat_path = "../data/for_gif/merged_images"

image_list = glob.glob(os.path.join(image_list_path, "*"))
mask_list = os.listdir("../data/for_gif/masks")
predict_list = glob.glob(os.path.join(predict_list_path, "*"))

for mask_name in tqdm(mask_list):
    
    img_name = mask_name[:-3]+"jpg" #images dosyasında isim karşılığını almak için uzantısını .jpg alıyoruz
    
    mask = cv2.imread(os.path.join(mask_list_path, mask_name))
    image = cv2.imread(os.path.join(image_list_path, img_name))
    predict = cv2.imread(os.path.join(predict_list_path, img_name))
    
    mask = cv2.resize(mask, (360, 227))
    image = cv2.resize(image, (360, 227))
    predict = cv2.resize(predict, (720, 453))
    
    concat_im = np.concatenate((image, mask), axis=1)
    
    concat_pred = np.concatenate((concat_im, predict), axis=0)
    cv2.imwrite(os.path.join(concat_path, img_name), concat_pred)
