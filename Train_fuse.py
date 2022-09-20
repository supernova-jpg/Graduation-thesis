import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils import data
import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from models.cls_model import Illumination_classifier
from models.common import gradient, clamp, init_seeds, RGB2YCrCb, YCrCb2RGB
from models.cls_model import Self_Encoder
import time

batch_size = 50
lr_init = 0.001

momentum = 0.9
weight_decay = 1e-4

to_tensor = transforms.Compose([transforms.ToTensor()])

parser = argparse.ArgumentParser()
makedir = lambda path: os.makedirs(path) if not os.path.exists(path) else None

for i in range(4):
    makedir('.\\pretrained\\'+str(i))

parser.add_argument('--vis_dataset_path', metavar='DIR', 
                    default='.\\Datasets\\Lp_Visible\\0',help='path to visible images') 

parser.add_argument('--ir_dataset_path', metavar='DIR', 
                    default='.\\Datasets\\Lp_Infrared\\0',help='where to load infrared images') 

parser.add_argument('--fusion_path', metavar='DIR', 
                    default='.\\Datasets\\Fusion_Result',help='where to load infrared images') 



args = parser.parse_args()



class MSRS_data(data.Dataset):
    def __init__(self, vis_dir, ir_dir, transform= to_tensor):
        super().__init__()
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.name_list = [y for y in os.listdir(vis_dir) if y in os.listdir(ir_dir)]
         
    
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # Obtain the name of current image
  
        inf_image = Image.open(os.path.join(self.ir_dir, name)).convert('L')  
        vis_image = Image.open(os.path.join(self.vis_dir, name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
      
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list)

if __name__ == '__main__':

    def train(vis_path, ir_path):

  
        epochs = 10


        min_loss = 1000
        train_dataset = MSRS_data(vis_path, ir_path)
        init_seeds(0)
        train_loader = DataLoader(
            train_dataset, batch_size= batch_size, shuffle=True,
            num_workers=1 , pin_memory=True)
    

        model_illum = Illumination_classifier(input_channels = 3)
        model_illum.load_state_dict(torch.load('./pretrained/best_cls.pth'))
        model_illum.eval()
     

        model_fuse = Self_Encoder()
    
   #train_tqdm = tqdm.tqdm(train_loader, total= epochs*len(train_loader))


        for epoch in range(1, epochs+1):
         
        
            if epoch < epochs // 2:
                lr = lr_init
            else:
                lr = lr_init * (epochs - epoch) / (epochs - epochs // 2)

            # The learning rate will be descended during the learning

            optimizer = optim.Adam(model_fuse.parameters(), lr= lr, weight_decay= weight_decay)

            for vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name in tqdm.tqdm(train_loader):

            # t = YCbCr2RGB2(vis_y_image[0], cb[0], cr[0])
            # transforms.ToPILImage()(t).save(name[0])
            
                output = model_illum(vis_image)
                day_p = output[:, 0]
                night_p = output[:, 1]
                vis_weight = torch.abs(day_p) / (torch.abs(day_p) + torch.abs(night_p))

            
                model_fuse.train()
                optimizer.zero_grad()
                pre_fused = model_fuse(vis_y_image, inf_image)
            
                pre_fused = clamp(pre_fused)
                
           
                loss =3*F.l1_loss(pre_fused*(1-vis_weight)[:, None, None, None],
                    inf_image*(1-vis_weight)[:, None, None, None]) + 3*F.l1_loss(
                        vis_weight[:, None, None, None] * pre_fused,
                        vis_weight[:, None, None, None] * vis_y_image) + 7* F.l1_loss(pre_fused, torch.max(0.7*vis_y_image, inf_image)) + 20*F.l1_loss(gradient(pre_fused), gradient(vis_y_image))
               
            

            #train_tqdm.set_postfix(epoch= epoch, loss = loss.item())

                loss.backward()
                optimizer.step()
            
                print('\nEpoch:{}, loss: {}'.format(epoch, loss.item()))


            if loss < min_loss:
                min_loss = loss
                best_epoch = epoch
                torch.save(model_fuse.state_dict(), f'.\\pretrained\\'+ str(i) + '\\Best_fusion_model.pth')

    time_start = time.process_time() 
    train('.\\Datasets\\Lp_Visible\\'+str(0),'.\\Datasets\\Lp_Infrared\\'+str(0))
    time_end = time.process_time() 
    print("Running time:",time_end - time_start)
    print('EXIT_SUCCESS!')
    #fused_image = YCrCb2RGB(pre_fused[0], vis_cb_image[0], vis_cr_image[0])
    #fused_image = transforms.ToPILImage()(fused_image)
