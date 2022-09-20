import os
import cv2

makedir = lambda path: os.makedirs(path) if not os.path.exists(path) else None

path_train = './Datasets/Lp_Visible/0/'
path_test = './Datasets/Test_dataset/Lp_Visible/0/'

def Separate(path):
    Day_Dir = path + 'Day/'
    Night_Dir = path + 'Night/'
    makedir(Day_Dir)
    makedir(Night_Dir)

    for filename in os.listdir(path):
    
        if 'D' in filename and 'png' in filename:
            img = cv2.imread(path + filename)
            day_img = img.copy()
           
            cv2.imwrite(Day_Dir+filename, day_img)

        if 'N' in filename and 'png' in filename:
            img = cv2.imread(path + filename)
            night_img = img.copy()
            cv2.imwrite(Night_Dir + filename, night_img)

Separate(path_train)
Separate(path_test)

print('EXIT_SUCCESS!')