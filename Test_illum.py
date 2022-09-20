import torch
from torchvision import transforms, datasets
from tqdm import tqdm

from models.cls_model import Illumination_classifier
import torch.nn.functional as F

model_illum = Illumination_classifier(input_channels=3)

batch_size = 100

test_dataset = datasets.ImageFolder(
        './Datasets/Test_dataset/Lp_Visible/0',
        transforms.Compose([
            transforms.ToTensor(),
        ]))

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size= batch_size, shuffle=True,
        pin_memory=True)


def test(model, test_loader):
    model.load_state_dict(torch.load('./pretrained/best_cls.pth', map_location= torch.device('cpu')))
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            output = model(data)
            day_p = output[:, 0]
            night_p = output[:, 1]
            vis_weight = torch.abs(day_p) / (torch.abs(day_p) + torch.abs(night_p))
            
            pred = output.data.max(1, keepdim=True)[1]
            test_loss += F.cross_entropy(output, target, size_average=False).item()  
            # sum up batch loss
            # get the index of the max log-probability
            print('Our prediction:'+str(vis_weight)+'\t'+'Target:'+str(target[0])+'\n')
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    prec1 = correct / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * prec1))
    return prec1

test(model_illum , test_loader)
