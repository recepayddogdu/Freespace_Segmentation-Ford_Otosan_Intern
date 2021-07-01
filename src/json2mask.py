import numpy as np
import cv2
import json
import os
import tqdm
from constant import *

if not os.path.exists(MASK_DIR): #MASK_DIR yolunda masks klasörü yoksa yeni klasör oluştur.
    os.mkdir(MASK_DIR)

jsons = os.listdir(JSON_DIR) #jsons klasöründeki dosyaların isimleri ile jsons listesi oluştur.

for json_name in tqdm.tqdm(jsons):    
    
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, "r") #Dosya oku
    json_dict = json.load(json_file) #Okunan dosyayı dict'e çevir
    
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    #Json dosyasındaki size'a göre zeros matrisi oluştur
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5]) #json_name'in sonundaki .json uzantısını silip yeni path oluştur
    
    
    for obj in json_dict["objects"]:
        if obj["classTitle"]=="Freespace":
            #freespace = np.array([obj["points"]["exterior"]], dtype=np.int32)
            cv2.fillPoly(mask, np.array([obj["points"]["exterior"]], dtype=np.int32), color=100)
            
            if obj["points"]["interior"] != []:
                for interior in obj["points"]["interior"]:
                    #fs_interior = np.array([interior], dtype=np.int32)
                    cv2.fillPoly(mask, np.array([interior], dtype=np.int32), color=0)

    cv2.imwrite(mask_path, mask.astype(np.uint8))