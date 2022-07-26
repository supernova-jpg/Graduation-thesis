from scipy.signal import wiener
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def Wiener(input):

    lenaNoise = input.astype('float64')
    lenaWiener = wiener(lenaNoise, [5,5])
    
    a = 255/(lenaWiener.max()-lenaWiener.min())
    b = -255*lenaWiener.min()/(lenaWiener.max()-lenaWiener.min())
    output = (a*lenaWiener+b).astype(np.uint8)
    return output

    
def read_path(file_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(os.listdir(file_pathname))
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        IM = cv2.split(img)
    
        for i in IM:
            i = Wiener(i) 
        
        img2 = cv2.merge(IM)
        img2 = cv2.fastNlMeansDenoisingColored(img2,None,5,5,7,21)

        #####save figure
        if 'train' in file_pathname:
            cv2.imwrite("./train/vi-modify/"+ filename,img2)  

        if 'test' in file_pathname:
            cv2.imwrite("./test/vi-modify/"+ filename,img2)  

read_path("C:\\Users\\Supernova\\Desktop\\Graduate_thesis\\MSRS\\train\\vi")
read_path("C:\\Users\\Supernova\\Desktop\\Graduate_thesis\\MSRS\\test\\vi")
