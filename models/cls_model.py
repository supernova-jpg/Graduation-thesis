import math

from torch import nn
import torch
from models.common import reflect_conv


class Illumination_classifier(nn.Module):
    def __init__(self, input_channels, init_weights=True):
        super(Illumination_classifier, self).__init__()

        self.conv1 = reflect_conv(in_channels=input_channels, out_channels=32)
        self.conv2 = reflect_conv(in_channels=32, out_channels=64)
        self.Pool = nn.MaxPool2d(2,stride =2)
        self.conv3 = reflect_conv(in_channels=64, out_channels=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride= 1, padding = 0)
    
        self.linear1 = nn.Linear(in_features=768, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=2)

      
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        activate = nn.LeakyReLU(inplace=True)
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.Pool(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.linear2(x)
      
        #x = nn.ReLU()(x)
        #x = nn.Softmax()(x)
        return x

    def _initialize_weights(self):
        """
        Initialling the weights

        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Self_Encoder(nn.Module):
    def __init__(self):
        super(Self_Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

        self.conv6 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv7 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv8 = reflect_conv(in_channels=32, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.conv9 = nn.Conv2d(in_channels=16, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        vi_out, ir_out = activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out))
        vi_out, ir_out = activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out))
        vi_out, ir_out = activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out))
        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))
        #print("\nvi shape:\t",vi_out.shape)
        #print("\nir shape:\t",ir_out.shape)
        #print("\nvis_weight shape:\t",vis_weight.shape)
        x = (vi_out + ir_out) /2.0
        
        x = activate(self.conv6(x))
        x = activate(self.conv7(x))
        x = activate(self.conv8(x))
        x = nn.Tanh()(self.conv9(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


