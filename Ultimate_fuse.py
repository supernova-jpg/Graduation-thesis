import argparse
import os
import cv2
from PIL import Image

import time
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.common import YCrCb2RGB, RGB2YCrCb, clamp
from models.cls_model import Self_Encoder
from Laplacian_Fusion import Reconstruct


batch_size = 1
ir_pyramid_path = 'Datasets/Lp_Infrared'
vis_pyramid_path = 'Datasets/Lp_Visible'

makedir = lambda path: os.makedirs(path) if not os.path.exists(path) else None
to_tensor = transforms.Compose([transforms.ToTensor()])

class MSRS_data(data.Dataset):

    def __init__(self, vis_dir, ir_dir, transform= to_tensor):
        super().__init__()
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.name_list = [y for y in os.listdir(vis_dir) if y in os.listdir(ir_dir)]
         # To obtain the filename in the folder
    
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # Obtain the name of current image
  
        inf_image = Image.open(os.path.join(self.ir_dir, name)).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vis_dir, name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
      
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--vis_img_path', metavar='DIR', default='Datasets/Lp_Visible/0')  
    parser.add_argument('--ir_img_path', metavar='DIR', default='Datasets/Lp_Infrared/0')  

    parser.add_argument('--save_path', default='Datasets\\Fusion_Result\\0')  
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--fusion_pretrained', default='pretrained\\0\\Best_fusion_model.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    

    args = parser.parse_args()
    Ultimate_Fuse = '.\\Datasets\\Fusion_Result\\Ultimate_fuse\\'
    makedir('Datasets\\Fusion_Result\\0')
    makedir('Datasets\\Fusion_Result\\1')
    makedir(Ultimate_Fuse)

    def implement(path1, path2, i):

        test_dataset = MSRS_data(path1, path2)
        test_loader = DataLoader(
            test_dataset, batch_size= batch_size, shuffle = True,
            num_workers=args.workers, pin_memory=True)

        model_fuse = Self_Encoder()
       
        model_fuse.load_state_dict(torch.load('pretrained\\'+str(i)+'\\Best_fusion_model.pth', map_location=torch.device('cpu')))

        model_fuse.eval()
        test_tqdm = tqdm(test_loader, total=len(test_loader))

        for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:

            # t = YCbCr2RGB2(vis_y_image[0], cb[0], cr[0])
            # transforms.ToPILImage()(t).save(name[0])
      
            fused_image = model_fuse(vis_y_image, inf_image)
            fused_image = clamp(fused_image)

            rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            #rgb_fused_image.save(f'{args.save_path}/{name[0]}')
            rgb_fused_image.save('Datasets\\Fusion_Result\\'+str(i) + '\\'+ name[0])

    time_start = time.process_time() 
    implement('Datasets/Lp_Visible/0','Datasets/Lp_Infrared/0',0)
    #implement('Datasets/Test_dataset/Lp_Visible/1','Datasets/Test_dataset/Lp_Infrared/1',1)


    for filename in os.listdir(ir_pyramid_path+'/0'):

        layer = []
        layer0 = cv2.imread('Datasets\\Fusion_Result\\0\\'+filename)
        layer.append(layer0)
        #layer1 = cv2.imread('Datasets\\Fusion_Result\\1\\'+filename)
        #layer.append(layer1)

        for i in range(1,4):
            ir = cv2.imread(ir_pyramid_path +'/'+str(i)+'/'+filename)
            vis = cv2.imread(vis_pyramid_path +'/'+str(i)+'/'+filename)
       
            fused = cv2.max(ir,vis)
            layer.append(fused)

        Fused_reconstruct = Reconstruct(layer)
        cv2.imwrite(Ultimate_Fuse+filename, Fused_reconstruct)
        print(filename+"\tCompleted")

    time_end = time.process_time() 
    print("Running time:",time_end - time_start)

        
print('EXIT_SUCCESS!')