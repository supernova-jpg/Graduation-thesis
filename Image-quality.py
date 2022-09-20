import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import math

ir_dir = ".\\Ori_Dataset\\train\\ir\\"
vis_dir = ".\\Ori_Dataset\\train\\vi-modify\\"
trad_fused_dir = ".\\Datasets\\Traditional_fused\\"
Siamese_dir = ".\\Datasets\\Siamese-network-Results\\"
Our_dir = ".\\Datasets\\Fusion_Result\\Ultimate_fuse\\"

def PSNR(file1, file2, file3):
    psnr1, psnr2 = [],[]
    for i in range (3):
        
        mse1 = np.mean((file1-file3)**2)
        mse2 = np.mean((file2-file3)**2)
        
        psnr1.append(10*np.log10(255*255/mse1))
        psnr2.append(10*np.log10(255*255/mse2))

    w = 0.7
    result = w*np.mean(psnr1)+ (1-w)*np.mean(psnr2)
    return result

def SD(image):
	image = image.astype(np.float32)
	m = image.shape[1]
	n = image.shape[0]
	u = np.mean(np.mean(image))
	SD = np.sqrt(np.sum(np.sum(np.clip(np.square(np.clip(image - u, 0, 255)), 0, 255))) / (m * n))
	return SD

def MSE(image, vi_image, ir_image):
	image = image.astype(np.float32) / 255.0
	vi_image = vi_image.astype(np.float32) / 255.0
	ir_image = ir_image.astype(np.float32) / 255.0
	[m, n, p] = image.shape
	MSE_AF = np.sum(np.sum(np.square(image - vi_image))) / (m * n)
	MSE_BF = np.sum(np.sum(np.square(image - ir_image))) / (m * n)
	MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
	return MSE


def MI(im1, im2):
	row, column, color = im1.shape
	count = row * column
	N = 256
	h = np.zeros((N, N))
	for i in range(row):
		for j in range(column):
			h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
	h = h / np.sum(h)
	im1_marg = np.sum(h, axis=0)
	im2_marg = np.sum(h, axis=1)
	H_x = 0
	H_y = 0
	for i in range(N):
		if (im1_marg[i] != 0):
			H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
	for i in range(N):
		if (im2_marg[i] != 0):
			H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
	H_xy = 0
	for i in range(N):
		for j in range(N):
			if (h[i, j] != 0):
				H_xy = H_xy + h[i, j] * math.log2(h[i, j])
	MI = H_xy - H_x - H_y
	return MI

def get_MI(image, ir_image, vi_image):
	MIA = MI(image, ir_image)
	MIB = MI(image, vi_image)
	MI_results = MIA + MIB
	return MI_results


PSNR_trad, PSNR_dl = [],[]

joint_path = [y for y in os.listdir(vis_dir) if y in os.listdir(ir_dir) and y in os.listdir(trad_fused_dir)and y in os.listdir(Siamese_dir)]

joint_path = joint_path[0:10]

PSNR_Eval = pd.DataFrame(columns = ['Traditional','Siamese_network','Deep_learning'])
MI_Eval = pd.DataFrame(columns = ['Traditional','Siamese_network','Deep_learning'])
SD_Eval = pd.DataFrame(columns = ['Traditional','Siamese_network','Deep_learning'])
MSE_Eval = pd.DataFrame(columns = ['Traditional','Siamese_network','Deep_learning'])


i = 0

for filename in joint_path:
    i +=1
    imread = lambda x: cv2.imread(x + filename)
    ir_file = imread(ir_dir)

    vis_file = imread(vis_dir)
    trad_fused_file = imread(trad_fused_dir)
    Siamese_file = imread(Siamese_dir)
    Our_file = imread(Our_dir)

    a = PSNR(ir_file, vis_file, trad_fused_file)
    b = PSNR(ir_file, vis_file, Siamese_file)
    c = PSNR(ir_file, vis_file, Our_file)
    #print('PSNR:\t'+str(a) + "\t"+ str(b) + "\t"+ str(c) + "\n")

    d = get_MI(trad_fused_file, ir_file, vis_file)
    e = get_MI(Siamese_file, ir_file, vis_file)
    f = get_MI(Our_file, ir_file, vis_file)
    #print('PSNR:\t'+str(d) + "\t"+ str(e) + "\t"+ str(f) + "\n")

    g = SD(trad_fused_file)
    h = SD(Siamese_file)
    i = SD(Our_file)

	
    j = MSE(trad_fused_file, ir_file, vis_file)
    k = MSE(Siamese_file, ir_file, vis_file)
    m = MSE(Our_file, ir_file, vis_file)

    PSNR_Eval.loc[i] = [a,b,c]
    MI_Eval.loc[i] = [d,e,f]
    SD_Eval.loc[i]= [g,h,i]
    MSE_Eval.loc[i] = [j,k,m]

PSNR1 = np.average(PSNR_Eval.Traditional)
PSNR2 = np.average(PSNR_Eval.Siamese_network)
PSNR3 = np.average(PSNR_Eval.Deep_learning)



MI1 = np.average(MI_Eval.Traditional)
MI2 = np.average(MI_Eval.Siamese_network)
MI3 = np.average(MI_Eval.Deep_learning)

SD1 = np.average(SD_Eval.Traditional)
SD2 = np.average(SD_Eval.Siamese_network)
SD3 = np.average(SD_Eval.Deep_learning)

MSE1 = np.average(MSE_Eval.Traditional)
MSE2 = np.average(MSE_Eval.Siamese_network)
MSE3 = np.average(MSE_Eval.Deep_learning)


PSNR_Eval.to_csv(".\\Evaluation\\PSNR.csv")
MI_Eval.to_csv(".\\Evaluation\\MI.csv")


#for i in range(len(joint_path)):
    #print(str(PSNR_trad[i]) + "\t"+ str(PSNR_dl[i]) + "\n")

print("Average" + str(PSNR1) + "\t"+ str(PSNR2) + "\t"+ str(PSNR3)+"\n")
print("Average" + str(MI1) + "\t"+ str(MI2) +  "\t"+ str(MI3)+"\n")
print("Average" + str(SD1) + "\t"+ str(SD2) +  "\t"+ str(SD3)+"\n")
print("Average" + str(MSE1) + "\t"+ str(MSE2) +  "\t"+ str(MSE3)+"\n")
print("EXIT_SUCCESS!")

