
import argparse
import os
import random

import numpy as np
from torch import optim
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.cls_model import Illumination_classifier

epochs = 30
batch_size = 128
initial_lr = 0.001
momentum = 0.9
image_size = 64

def test(model, test_loader):
    # test
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    prec1 = correct / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * prec1))
    return prec1


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='./Datasets/Lp_Visible/0',
                        help='path to dataset')
    parser.add_argument('--save_path', default='./pretrained/')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
   

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    train_dataset = datasets.ImageFolder(
        args.dataset_path,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    # Divide the data into validation and training data
    image_nums = len(train_dataset)
    train_nums = int(image_nums * 0.9)
    test_nums = image_nums - train_nums
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=train_dataset, lengths=[train_nums, test_nums])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size= batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size= batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    model = Illumination_classifier(input_channels=3)
        #model = model.cuda()
    optimizer = optim.Adam(model.parameters(), 
    lr= initial_lr,weight_decay=args.weight_decay)
    best_prec1 = 0.0
    for epoch in range(args.start_epoch, epochs):
    
        # We self-defined the descendant of our learning rate.The first half of learning rate
        if epoch < epochs // 2:
            lr = initial_lr
        else:
            lr = initial_lr * (epochs - epoch) / (epochs - epochs // 2)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
    
        print('{:d}/{:d}'.format(epoch, (epochs-args.start_epoch)))
        train_tqdm = tqdm(train_loader, total=len(train_loader))
        # If it is in the daytime, the one-hot label is[1,0], while during the night is [0,1]
        for image, label in train_tqdm:
    
                optimizer.zero_grad()
                output = model(image) # This is the time when we see the magic happens!!!
                loss = F.cross_entropy(output, label) # Loss function
                train_tqdm.set_postfix(epoch=epoch, loss_total=loss.item())
                loss.backward()
                optimizer.step()

        prec1 = test(model, test_loader)
        if best_prec1 < prec1:
            torch.save(model.state_dict(), f'{args.save_path}/best_cls.pth')
            best_prec1 = prec1

    print("EXIT_SUCCESS!")


