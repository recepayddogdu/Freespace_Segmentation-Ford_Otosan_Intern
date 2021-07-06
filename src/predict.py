import os
import glob
import torch
import tqdm
from preprocess import tensorize_image
import numpy as np
from constant import *
from train import *

#### PARAMETERS #####
cuda = False
model_name = "Unet_1.pt"
PREDICT_DIR = PREDICT_DIR + "/" + model_name.split(".")[0]
model_path = os.path.join(MODELS_DIR, model_name)
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
    
    for image in tqdm.tqdm(images):
        batch_test = tensorize_image([image], input_shape, cuda)
        output = model(batch_test)
        out = torch.argmax(output, axis=1)
        
        if cuda:
            out = out.cuda()
        else:
            out = out.cpu()
        
        outputs_list  = out.detach().numpy()
        mask = np.squeeze(outputs_list, axis=0)
        
        img = cv2.imread(batch_test[0])
        img_resize = cv2.resize(img, input_shape)
        mask_ind = mask == 1
        copy_img = img_resize.copy()
        img_resize[mask==1, :] = (255, 0, 125)
        opac_image = (img_resize/2 + copy_img/2).astype(np.uint8)
        cv2.imwrite(os.path.join(PREDICT_DIR, batch_test.split("/")[-1]), opac_image)
        
if __name__ == "__main__":
    predict(model, test_input_path_list)



















