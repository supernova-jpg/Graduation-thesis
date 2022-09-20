import cv2  
import os

level = 4


makedir = lambda path: os.makedirs(path) if not os.path.exists(path) else None

def path_selection(mode):
    if mode == 'train':
        vis_dataset_path = '.\\Ori_Dataset\\train\\vi-modify'
        ir_dataset_path = '.\\Ori_Dataset\\train\\ir'

    elif mode == 'test':
        vis_dataset_path = '.\\Ori_Dataset\\test\\vi-modify'
        ir_dataset_path = '.\\Ori_Dataset\\test\\ir'
    
    return vis_dataset_path, ir_dataset_path


def sameSize(img1, img2):   
    rows, cols, dpt = img2.shape  
    dst = img1[:rows,:cols]  
    return dst  
 
def Decompose(ir_path, vis_path, mode):

    joint_path = [y for y in os.listdir(vis_path) if y in os.listdir(ir_path)]
    
    for filename in joint_path:

        visible = cv2.imread(vis_path+'/'+filename)
        infrared = cv2.imread(ir_path+'/'+filename)

        G = visible.copy()
        R = infrared.copy()  

        gp_visible = Gauss_decompose(G)
        gp_infrared = Gauss_decompose(R)

        lp_visible = Laplace(gp_visible)
        lp_infrared = Laplace(gp_infrared)

        for i in range(len(lp_visible)):
            if mode == 'test':
                cv2.imwrite("./Datasets/Test_dataset/Lp_Visible/" + str(i) + '/'+ filename,lp_visible[i])
            else:
                cv2.imwrite("./Datasets/Lp_Visible/" + str(i) + '/'+ filename,lp_visible[i])
        for i in range(len(lp_infrared)):
            if mode == 'test':
                cv2.imwrite("./Datasets/Test_dataset/Lp_Infrared/" + str(i) + '/'+ filename,lp_infrared[i])
            else:
                cv2.imwrite("./Datasets/Lp_Infrared/" + str(i) + '/'+ filename,lp_infrared[i])
        
        print(filename +"\t"+"Completed")
    return None

def Gauss_decompose(graph):
    gp = [graph]
    for i in range(level):  
        graph = cv2.pyrDown(graph)  
        gp.append(graph)  
    return gp

def Laplace(gp):
    lp = [gp[level-1]]  # Laplacian pyramid

    for i in range(level-1,0,-1):  
        GE = cv2.pyrUp(gp[i]) 
        L = cv2.subtract(gp[i-1],GE)  
     
        lp.append(L)  
    return lp
    
modes = ['train','test']

makedir('./Datasets/Test_dataset/Lp_Infrared')
makedir('./Datasets/Test_dataset/Lp_Visible')

for i in range(0,4):
    makedir('./Datasets/Test_dataset/Lp_Infrared/'+str(i))
    makedir('./Datasets/Test_dataset/Lp_Visible/'+str(i))


makedir('./Datasets/Lp_Infrared')
makedir('./Datasets/Lp_Visible')

for i in range(0,4):
    makedir('./Datasets/Lp_Infrared/'+str(i))
    makedir('./Datasets/Lp_Visible/'+str(i))

for mode in modes:
    vis_dataset_path, ir_dataset_path = path_selection(mode)
    Decompose(ir_dataset_path,vis_dataset_path,mode)

print('EXIT_SUCCESS!')