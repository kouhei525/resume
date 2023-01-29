import glob
import random
import os
import numpy as np
import time
import datetime
import sys

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from visdom import Visdom

import itertools
from PIL import Image

import matplotlib.pyplot as plt

def main():

    class ResidualBlock(nn.Module):
        def __init__(self, in_features):
            super(ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features)
            )

        def forward(self, x):
            return x + self.conv_block(x)

    class Generator(nn.Module):
        def __init__(self, input_nc, output_nc, n_residual_blocks=9):
            super(Generator, self).__init__()

            self.model = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),

                ResidualBlock(256),
                ResidualBlock(256),
                ResidualBlock(256),
                ResidualBlock(256),
                ResidualBlock(256),
                ResidualBlock(256),
                ResidualBlock(256),
                ResidualBlock(256),
                ResidualBlock(256),

                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),

                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, 7),
                nn.Tanh()
            )

        def forward(self, x):
            return self.model(x)


    class Opts():
        def __init__(self):
            self.start_epoch = 0
            self.n_epochs = 5
            self.batch_size = 1
            self.dataroot = '/content/gdrive/MyDrive/gan_sample-main/chapter5/photo2portrait'
            self.lr = 0.0002
            self.decay_epoch = 200
            self.size = 256
            self.input_nc = 3
            self.output_nc = 3
            self.cpu = False
            self.n_cpu = 8
            self.device_name = "cuda:0"
            self.device = torch.device(self.device_name)
            self.load_weight = False

    opt = Opts()

    transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                    transforms.RandomCrop(opt.size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    #Tensor = torch.cuda.FloatTensor if not opt.cpu else torch.Tensor
    Tensor = torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
    netG_A2B = Generator(opt.input_nc, opt.output_nc)


    netG_A2B.load_state_dict(torch.load("netG_A2B.pth", map_location="cpu"), strict=False)

    transform_ = transforms.Compose(transforms_)

    item_A = transform_(Image.open("./static/camera_capture.jpg").convert('RGB'))
    real_A = Variable(input_A.copy_(item_A))

    netG_A2B = netG_A2B.to("cpu")

    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)

    save_image(fake_B, "static/fake_B.png")
    #img = np.array(fake_B.detach().numpy())
    #print(np.array(fake_B.detach().numpy()))
    #print(img.shape)
    #img = plt.imread("static/fake_B.png")
    #plt.imshow(img)
    #plt.show()

if __name__ == "__main__":
    main()