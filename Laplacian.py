import cv2  
import numpy as np  
from matplotlib import pyplot as plt  
import os
 
level = 4

def sameSize(img1, img2):   
    rows, cols, dpt = img2.shape  
    dst = img1[:rows,:cols]  
    return dst  
 
def read_path(ir_path, vis_path):

    joint_path = [y for y in os.listdir(vis_path) if y in os.listdir(ir_path)]
    
    for filename in joint_path:

        visible = cv2.imread(vis_path+'/'+filename)
        infrared = cv2.imread(ir_path+'/'+filename)

        G = visible.copy()
        R = infrared.copy()  

        gp_visible = Decompose(G)
        gp_infrared = Decompose(R)

        lp_visible = Laplace(gp_visible)
        lp_infrared = Laplace(gp_infrared)

        fused_layer = []
        weights = [0.9,0.4,0.2,0.9]

        for i in range(level):
            layer = cv2.addWeighted(lp_visible[i], weights[i], lp_infrared[i],1-weights[i],0)
            fused_layer.append(layer)

        Fused_reconstruct = Reconstruct(fused_layer)

        
        cv2.imwrite("./train/vi-modify/"+ filename,Fused_reconstruct)  
        print(filename)




def Decompose(graph):
    gp = [graph]
    for i in range(level):  
        graph = cv2.pyrDown(graph)  
        gp.append(graph)  
    return gp

def Laplace(gp):
    lp = [gp[level-1]]  # Laplacian pyramid

    for i in range(level-1,0,-1):  
        GE = cv2.pyrUp(gp[i]) 
        L = cv2.subtract(gp[i-1], sameSize(GE,gp[i-1]))  
        # cv2.imshow('Laplacian layer'+str(i),L) 
        lp.append(L)  
    return lp


def Reconstruct(pyramid):
    layer = pyramid[0]

    for i in range(1,level):  
        layer = cv2.pyrUp(layer)  
        layer = cv2.addWeighted(sameSize(layer, pyramid[i]),1,pyramid[i],2,0)  
    return layer

read_path("C:\\Users\\Supernova\\Desktop\\Graduate_thesis\\MSRS\\train\\ir",
"C:\\Users\\Supernova\\Desktop\\Graduate_thesis\\MSRS\\train\\vi-modify")







