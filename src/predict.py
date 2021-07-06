import os
import glob
import torch
import tqdm
from tqdm import tqdm_notebook
from preprocess import tensorize_image
import numpy as np
from constant import *
from train import *

if not os.path.exists(PREDICT_DIR): #PREDICT_DIR yolunda predicts klasörü yoksa yeni klasör oluştur.
    os.mkdir(PREDICT_DIR)

#### PARAMETERS #####
cuda = True
model_name = "Unet_1.pt"

predict_path = os.path.join(PREDICT_DIR, model_name.split(".")[0])
if not os.path.exists(predict_path): #predict_path yolunda predicts klasörü yoksa yeni klasör oluştur.
    os.mkdir(predict_path)

model_path = os.path.join(MODELS_DIR, model_name)
input_shape = input_shape
#####################

# LOAD MODEL
model = torch.load(model_path)
#Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
#Failing to do this will yield inconsistent inference results.
model.eval()

if cuda:
    model = model.cuda()



# PREDICT
def predict(model, images):


    for image in tqdm_notebook(images):
        batch_test = tensorize_image([image], input_shape, cuda)
        output = model(batch_test)
        out = torch.argmax(output, axis=1)
        
        
        out = out.cpu()
        
        outputs_list  = out.detach().numpy()
        mask = np.squeeze(outputs_list, axis=0)
       
        mask_uint8 = mask.astype('uint8')
        mask_resize = cv2.resize(mask_uint8, (1920, 1208), interpolation = cv2.INTER_AREA)
        
        img = cv2.imread(image)
        img_resize = cv2.resize(img, input_shape)
        mask_ind = mask_resize == 1
        #copy_img = img_resize.copy()
        copy_img = img.copy()
        img[mask_resize==1, :] = (255, 0, 125)
        opac_image = (img/2 + copy_img/2).astype(np.uint8)
        cv2.imwrite(os.path.join(predict_path, image.split("/")[-1]), opac_image)
        print("mask size from model: ", mask.shape),
        print("resized mask size: ", mask_resize.shape)
if __name__ == "__main__":
    predict(model, test_input_path_list)
    