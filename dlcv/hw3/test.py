import numpy as np
from skimage import io 
import scipy
import os
import sys 
from keras.models import load_model
from my_classes import DataGenerator
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-m', dest='model',type=str,required=True)
args = parser.parse_args()

index2rgb = {}
index2rgb[0] = [0,255,255]
index2rgb[1] = [255,255,0]
index2rgb[2] = [255,0,255]
index2rgb[3] = [0,255,0]
index2rgb[4] = [0,0,255]
index2rgb[5] = [255,255,255]
index2rgb[6] = [0,0,0]

def lable2rgb(filepath,model):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    for file in file_list:
        #if(file!='0013_sat.jpg'): continue
        read_path = os.path.join(filepath,file) 
        print(read_path)
        img = scipy.misc.imread(read_path).reshape(1,512,512,3)/255
        result = model.predict(img)       
        #for i in range(5):
             #print(result[0,0,i])
        #print(result[:5])                                                                 
        mask = np.argmax(result,axis=3).reshape(512,512,1)
        #print(mask.shape)
        #input()
        mask = 1 * mask[:,:,0]
        masks = np.empty((512,512,3),dtype='uint8')
        masks[mask == 0] = index2rgb[0]  # (Cyan: 011) Urban land                                                            
        masks[mask == 1] = index2rgb[1]  # (Yellow: 110) Agriculture land                                                          
        masks[mask == 2] = index2rgb[2]  # (Purple: 101) Rangeland                                                                  
        masks[mask == 3] = index2rgb[3]  # (Green: 010) Forest land                                                                
        masks[mask == 4] = index2rgb[4]  # (Blue: 001) Water                                                                        
        masks[mask == 5] = index2rgb[5]  # (White: 111) Barren land                                                                 
        masks[mask == 6] = index2rgb[6]  # (Black: 000) Unknown  
        #print(masks)
        img = Image.fromarray(masks, 'RGB')
        img.save('eval_p/'+file[:5]+'mask.png')
        
model = load_model(args.model)
lable2rgb('validation',model)

print('evaluating ...')
os.system('python mean_iou_evaluate.py  -g validation/ -p eval_p/')
