from scipy.signal import wiener
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)    

def Wiener(input):

    lenaNoise = input.astype('float64')
    lenaWiener = wiener(lenaNoise, [5,5])
    
    a = 255/(lenaWiener.max()-lenaWiener.min())
    b = -255*lenaWiener.min()/(lenaWiener.max()-lenaWiener.min())
    output = (a*lenaWiener+b).astype(np.uint8)
    return output

    
def read_path(file_pathname):

    for filename in os.listdir(file_pathname):

        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        IM = cv2.split(img)
    
        for i in IM:
            i = Wiener(i) 
        
        img2 = cv2.merge(IM)
        img2 = cv2.fastNlMeansDenoisingColored(img2,None,5,5,7,21)

        #####save figure
        if 'train' in file_pathname:
            cv2.imwrite("./Ori_Dataset/train/vi-modify/"+ filename,img2)  

        if 'test' in file_pathname:
            cv2.imwrite("./Ori_Dataset/test/vi-modify/"+ filename,img2)  
        
        

makedir("./Ori_Dataset/train/vi-modify/")
makedir("./Ori_Dataset/test/vi-modify/")

read_path("./Ori_Dataset/train/vi")
read_path("./Ori_Dataset/test/vi")
print("EXIT_SUCCESS!")