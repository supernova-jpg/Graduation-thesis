import cv2  
import os
import argparse

 
level = 4

makedir = lambda path: os.makedirs(path) if not os.path.exists(path) else None

vis_dataset_path_train, vis_dataset_path_test = './Datasets/Lp_Visible', './Datasets/Test_dataset/Lp_Visible'
ir_dataset_path_train, ir_dataset_path_test = './Datasets/Lp_Infrared', './Datasets/Test_dataset/Lp_Infrared'

parser = argparse.ArgumentParser(description='PyTorch PIAFusion')


parser.add_argument('--output_dataset_path', metavar='DIR', 
                    default='./Datasets/Traditional_fused',help='Where the output fused images storage in') 

args = parser.parse_args()

def sameSize(img1, img2):   
    rows, cols, dpt = img2.shape  
    dst = img1[:rows,:cols]  
    return dst  
 
def execute(ir_path, vis_path):

    joint_list = [y for y in os.listdir(vis_path +'/'+ str(0)) if y in os.listdir(ir_path +'/'+ str(0))]
    weights = 0.7
    
    path_test = './Datasets/Traditional_fused_Test'
    makedir(path_test)
    path_train = './Datasets/Traditional_fused'
    makedir(path_train)
    path_reconstruct = './Datasets/Reconstruct'
    makedir(path_reconstruct)

    for filename in joint_list:
        Vis_layer, Ir_layer, fused_layer = [],[],[]

        for i in range(level):
            visible = cv2.imread(vis_path +'/'+ str(i)+ '/'+filename)
            infrared= cv2.imread(ir_path +'/'+ str(i)+ '/'+filename)
            #infrared = cv2.IMREAD_GRAYSCALE(ir_path+'/'+ str(i)+'/'+filename)
            #hsv = cv2.cvtColor(visible, cv2.COLOR_BGR2HSV)
            #hsv = hsv[0]

            Vis_layer.append(visible)
            Ir_layer.append(infrared)

        for i in range(level):
            layer = cv2.addWeighted(Vis_layer[i], weights, Ir_layer[i],1-weights,0)
            fused_layer.append(layer)

        Fused_reconstruct = Reconstruct(fused_layer)
        #if 'Test' in vis_path:
            #cv2.imwrite(path_test +'/' +filename,Fused_reconstruct)  
        #else:
            #cv2.imwrite(path_train +'/' +filename,Fused_reconstruct) 
        cv2.imwrite(path_reconstruct +'/' +filename,Fused_reconstruct)

        print(filename+"\tCompleted")


def Reconstruct(pyramid):
    layer = pyramid[0]

    for i in range(1,level):  
        layer = cv2.pyrUp(layer)  
        layer = cv2.addWeighted(sameSize(layer, pyramid[i]),1,pyramid[i],2,0)  
    return layer

if __name__ == '__main__':
    
    execute(ir_dataset_path_train,vis_dataset_path_train)
    execute(ir_dataset_path_test,vis_dataset_path_test)
    print("EXIT_SUCCESS!")





