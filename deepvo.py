USE_TPU=False

import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from image_parameters import image_size
from torchvision import transforms
from torch.autograd import Variable

from azure.storage.blob import ContainerClient
import numpy as np
import io
import cv2
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# %matplotlib inline
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import pandas as pd
import gc


import io
import os
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import argparse
import torch


import os.path
import random
import datetime

import torch.utils.data
import torchvision.transforms as transforms

# Dataset website: http://theairlab.org/tartanair-dataset/
account_url = 'https://tartanair.blob.core.windows.net/'
container_name = 'tartanair-release1'

container_client = ContainerClient(account_url=account_url, 
                                 container_name=container_name,
                                 credential=None)
from tartan_azure_helpers import *


train_list, val_list, test_list = ([#'hospital/Hard/P040/',
  # 'carwelding/Hard/P001/',
  # 'soulcity/Hard/P002/',
  # 'abandonedfactory_night/Hard/P011/',
  # 'gascola/Easy/P003/',
  # 'endofworld/Easy/P008/',
  # 'hospital/Easy/P017/',
  # 'carwelding/Easy/P006/',
  # 'westerndesert/Easy/P010/',
  # 'westerndesert/Easy/P008/',
  # 'seasonsforest_winter/Easy/P005/',
  # 'seasonsforest_winter/Easy/P002/',
  # 'hospital/Hard/P042/',
  # 'oldtown/Easy/P002/',
  # 'office2/Hard/P010/',
  # 'westerndesert/Easy/P011/',
  # 'seasonsforest/Hard/P001/',
  # 'gascola/Hard/P001/',
  # 'amusement/Hard/P000/',
  # 'office2/Hard/P001/',
  # 'abandonedfactory/Hard/P009/',
  # 'gascola/Hard/P003/',
  # 'gascola/Hard/P008/',
  # 'abandonedfactory_night/Hard/P005/',
  # 'abandonedfactory_night/Easy/P009/',
  # 'westerndesert/Hard/P001/',
  # 'abandonedfactory_night/Hard/P003/',
  # 'neighborhood/Hard/P012/',
  # 'hospital/Easy/P000/',
  # 'amusement/Hard/P007/',
  # 'seasidetown/Easy/P005/',
  # 'hospital/Easy/P032/',
  # 'seasonsforest_winter/Easy/P001/',
  # 'office/Hard/P004/',
  # 'hospital/Easy/P023/',
  # 'neighborhood/Easy/P001/',
  # 'abandonedfactory_night/Hard/P006/',
  # 'endofworld/Easy/P000/',
  # 'office2/Hard/P009/',
  # 'hospital/Easy/P007/',
  # 'westerndesert/Easy/P007/',
  # 'abandonedfactory_night/Easy/P006/',
  # 'gascola/Easy/P007/',
  # 'westerndesert/Easy/P005/',
  # 'seasonsforest_winter/Easy/P007/',
  # 'abandonedfactory_night/Hard/P010/',
  # 'hospital/Easy/P009/',
  # 'hospital/Easy/P012/',
  # 'ocean/Easy/P009/',
  # 'abandonedfactory/Easy/P000/',
  # 'seasonsforest_winter/Easy/P008/',
  # 'office/Easy/P002/',
  # 'soulcity/Hard/P001/',
  # 'westerndesert/Hard/P006/',
#   'seasonsforest_winter/Easy/P003/',
#   'abandonedfactory/Hard/P010/',
#   'westerndesert/Easy/P006/',
#   'abandonedfactory/Easy/P011/',
#   'hospital/Hard/P046/',
#   'ocean/Hard/P006/',
#   'hospital/Easy/P026/',
#   'hospital/Hard/P037/',
#   'soulcity/Hard/P000/',
#   'neighborhood/Hard/P003/',
#   'neighborhood/Easy/P021/',
#   'hospital/Easy/P003/',
#   'carwelding/Easy/P007/',
#   'office/Hard/P005/',
#   'ocean/Hard/P009/',
#   'office/Easy/P004/',
#   'office2/Hard/P003/',
#   'seasidetown/Easy/P001/',
#   'ocean/Hard/P008/',
#   'office2/Easy/P007/',
#   'abandonedfactory_night/Easy/P008/',
#   'japanesealley/Easy/P002/',
#   'neighborhood/Easy/P002/',
#   'amusement/Hard/P006/',
#   'japanesealley/Easy/P004/',
#   'neighborhood/Easy/P004/',
#   'office2/Easy/P011/',
#   'abandonedfactory_night/Easy/P002/',
#   'neighborhood/Hard/P006/',
#   'hospital/Easy/P015/',
#   'abandonedfactory_night/Hard/P002/',
#   'seasonsforest_winter/Hard/P015/',
#   'seasonsforest/Hard/P002/',
#   'westerndesert/Easy/P009/',
#   'neighborhood/Hard/P016/',
#   'seasidetown/Easy/P002/',
#   'carwelding/Easy/P002/',
#   'abandonedfactory_night/Easy/P011/',
#   'office/Hard/P003/',
#   'soulcity/Easy/P008/',
#   'gascola/Hard/P000/',
#   'neighborhood/Hard/P015/',
#   'hospital/Easy/P014/',
#   'abandonedfactory/Hard/P011/',
#   'neighborhood/Hard/P013/',
#   'ocean/Easy/P006/',
#   'hospital/Hard/P047/',
#   'japanesealley/Easy/P007/',
#   'oldtown/Hard/P004/',
#   'neighborhood/Easy/P013/',
#   'ocean/Hard/P005/',
#   'seasonsforest/Easy/P010/',
#   'soulcity/Easy/P006/',
#   'abandonedfactory/Easy/P002/',
#   'seasonsforest_winter/Hard/P011/',
#   'office/Easy/P000/',
#   'soulcity/Easy/P010/',
#   'office/Hard/P002/',
#   'hospital/Easy/P004/',
#   'hospital/Easy/P035/',
#   'neighborhood/Hard/P004/',
#   'japanesealley/Easy/P003/',
#   'soulcity/Easy/P005/',
#   'soulcity/Hard/P009/',
#   'soulcity/Easy/P007/',
#   'abandonedfactory/Hard/P005/',
#   'hospital/Easy/P005/',
#   'abandonedfactory_night/Easy/P003/',
#   'seasonsforest_winter/Hard/P014/',
#   'abandonedfactory_night/Easy/P004/',
#   'hospital/Hard/P048/',
#   'carwelding/Easy/P005/',
#   'neighborhood/Easy/P007/',
#   'ocean/Easy/P000/',
#   'hospital/Easy/P019/',
#   'hospital/Easy/P031/',
#   'office2/Hard/P002/',
#   'abandonedfactory/Hard/P000/',
#   'hospital/Easy/P028/',
#   'seasonsforest/Easy/P009/',
#   'soulcity/Easy/P011/',
#   'oldtown/Hard/P007/',
#   'seasonsforest_winter/Easy/P004/',
#   'seasonsforest/Hard/P004/',
#   'ocean/Hard/P000/',
#   'seasonsforest_winter/Hard/P012/',
#   'office/Easy/P001/',
#   'seasonsforest_winter/Hard/P013/',
#   'hospital/Easy/P030/',
#   'ocean/Easy/P001/',
#   'japanesealley/Hard/P004/',
#   'office/Easy/P003/',
#   'abandonedfactory/Easy/P006/',
#   'hospital/Easy/P036/',
#   'neighborhood/Hard/P001/',
#   'ocean/Hard/P007/',
#   'abandonedfactory/Hard/P006/',
#   'ocean/Easy/P005/',
#   'oldtown/Hard/P006/',
#   'westerndesert/Easy/P001/',
#   'japanesealley/Hard/P005/',
#   'carwelding/Hard/P003/',
#   'abandonedfactory_night/Hard/P009/',
#   'endofworld/Easy/P009/',
#   'neighborhood/Easy/P017/',
#   'amusement/Easy/P002/',
#   'seasidetown/Hard/P003/',
#   'abandonedfactory_night/Easy/P007/',
#   'soulcity/Hard/P003/',
#   'abandonedfactory_night/Easy/P012/',
#   'neighborhood/Easy/P020/',
#   'endofworld/Hard/P002/',
#   'gascola/Hard/P009/',
#   'ocean/Easy/P013/',
#   'endofworld/Hard/P000/',
#   'amusement/Easy/P003/',
#   'seasonsforest_winter/Easy/P000/',
#   'carwelding/Easy/P001/',
#   'abandonedfactory_night/Hard/P007/',
#   'abandonedfactory/Easy/P010/',
#   'hospital/Easy/P025/',
#   'abandonedfactory/Easy/P009/',
#   'westerndesert/Easy/P012/',
#   'office/Easy/P005/',
#   'seasonsforest/Hard/P006/',
#   'oldtown/Easy/P001/',
#   'soulcity/Easy/P003/',
#   'office2/Easy/P009/',
#   'gascola/Easy/P001/',
#   'oldtown/Easy/P000/',
#   'hospital/Hard/P044/',
#   'hospital/Easy/P008/',
#   'abandonedfactory/Easy/P004/',
#   'hospital/Easy/P027/',
#   'hospital/Easy/P024/',
#   'endofworld/Easy/P003/',
#   'neighborhood/Easy/P009/',
#   'hospital/Easy/P013/',
#   'seasidetown/Hard/P002/',
#   'gascola/Easy/P006/',
#   'neighborhood/Hard/P007/',
#   'hospital/Hard/P045/',
#   'seasidetown/Easy/P004/',
#   'seasidetown/Hard/P001/',
#   'hospital/Easy/P029/',
#   'endofworld/Easy/P002/',
#   'office2/Hard/P006/',
#   'japanesealley/Easy/P001/',
#   'abandonedfactory_night/Hard/P012/',
#   'hospital/Hard/P043/',
#   'japanesealley/Hard/P001/',
#   'office2/Easy/P003/',
#   'office2/Hard/P008/',
#   'carwelding/Hard/P000/',
#   'carwelding/Easy/P004/',
#   'oldtown/Easy/P007/',
#   'endofworld/Hard/P005/',
#   'office2/Hard/P005/',
#   'office2/Easy/P005/',
#   'soulcity/Easy/P000/',
#   'gascola/Hard/P004/',
#   'neighborhood/Hard/P008/',
#   'abandonedfactory/Easy/P008/',
#   'ocean/Easy/P012/',
#   'soulcity/Easy/P004/',
#   'seasidetown/Easy/P006/',
#   'westerndesert/Easy/P013/',
#   'abandonedfactory_night/Easy/P010/',
#   'hospital/Easy/P033/',
#   'neighborhood/Hard/P000/',
#   'ocean/Hard/P003/',
#   'amusement/Hard/P003/',
#   'abandonedfactory_night/Hard/P008/',
#   'carwelding/Hard/P002/',
#   'neighborhood/Easy/P005/',
#   'seasonsforest_winter/Easy/P009/',
#   'endofworld/Easy/P001/',
#   'seasonsforest_winter/Easy/P006/',
#   'oldtown/Easy/P005/',
#   'abandonedfactory/Hard/P003/',
#   'oldtown/Hard/P005/',
#   'seasonsforest/Easy/P007/',
#   'oldtown/Hard/P001/',
#   'abandonedfactory_night/Hard/P000/',
#   'ocean/Easy/P011/',
#   'neighborhood/Hard/P009/',
#   'neighborhood/Easy/P018/',
#   'seasidetown/Easy/P000/',
#   'neighborhood/Hard/P017/',
#   'abandonedfactory_night/Easy/P013/',
#   'soulcity/Easy/P002/',
#   'hospital/Easy/P006/',
#   'westerndesert/Hard/P002/',
#   'abandonedfactory_night/Hard/P001/',
#   'ocean/Easy/P010/',
#   'office2/Easy/P008/',
#   'endofworld/Hard/P001/',
#   'office/Hard/P007/',
#   'office/Easy/P006/',
#   'hospital/Hard/P041/',
#   'neighborhood/Easy/P015/',
#   'hospital/Easy/P010/',
#   'abandonedfactory/Hard/P007/',
#   'oldtown/Hard/P008/',
  'endofworld/Easy/P006/',
  'amusement/Easy/P004/',
  'seasonsforest_winter/Hard/P017/',
  'ocean/Hard/P001/',
  'westerndesert/Hard/P000/',
  'neighborhood/Easy/P016/',
  'ocean/Easy/P008/',
  'hospital/Easy/P034/',
  'soulcity/Easy/P001/',
  'seasidetown/Hard/P000/',
  'office/Hard/P001/',
  'office/Hard/P006/',
  'japanesealley/Easy/P005/',
  'office2/Hard/P007/',
  'abandonedfactory_night/Hard/P013/',
  'hospital/Easy/P021/',
  'oldtown/Hard/P002/',
  'soulcity/Easy/P009/',
  'amusement/Hard/P005/',
  'gascola/Hard/P005/',
  'neighborhood/Hard/P002/',
  'neighborhood/Hard/P011/',
  'neighborhood/Hard/P010/',
  'neighborhood/Easy/P008/',
  'gascola/Easy/P004/',
  'office2/Easy/P006/',
  'oldtown/Hard/P000/',
  'office2/Easy/P004/',
  'seasonsforest_winter/Hard/P016/',
  'seasidetown/Easy/P003/',
  'soulcity/Hard/P008/',
  'hospital/Hard/P038/',
  'seasonsforest/Easy/P008/',
  'endofworld/Easy/P007/',
  'westerndesert/Hard/P004/',
  'westerndesert/Hard/P003/',
  'seasidetown/Easy/P009/',
  'gascola/Hard/P007/',
  'hospital/Easy/P022/',
  'ocean/Easy/P004/',
  'westerndesert/Hard/P005/',
  'abandonedfactory_night/Hard/P014/',
  'ocean/Hard/P004/',
  'oldtown/Easy/P004/',
  'ocean/Hard/P002/',
  'amusement/Easy/P001/',
  'seasonsforest/Hard/P005/',
  'endofworld/Easy/P005/',
  'seasidetown/Easy/P008/',
  'neighborhood/Hard/P014/',
  'hospital/Easy/P011/',
  'neighborhood/Easy/P014/',
  'abandonedfactory/Hard/P008/',
  'japanesealley/Hard/P000/',
  'soulcity/Easy/P012/',
  'endofworld/Easy/P004/',
  'abandonedfactory/Hard/P004/',
  'seasidetown/Hard/P004/',
  'abandonedfactory/Easy/P005/',
  'hospital/Hard/P049/',
  'westerndesert/Hard/P007/',
  'neighborhood/Easy/P003/',
  'seasonsforest/Easy/P001/',
  'amusement/Easy/P006/',
  'neighborhood/Easy/P019/',
  'neighborhood/Easy/P012/',
  'soulcity/Hard/P004/',
  'ocean/Easy/P002/',
  'neighborhood/Easy/P000/',
  'neighborhood/Easy/P010/',
  'amusement/Hard/P001/',
  'seasidetown/Easy/P007/',
  'hospital/Easy/P002/',
  'amusement/Easy/P008/',
  'abandonedfactory/Hard/P001/',
  'japanesealley/Hard/P002/',
  'seasonsforest/Easy/P002/',
  'seasonsforest/Easy/P004/',
  'seasonsforest/Easy/P011/',
  'hospital/Easy/P016/',
  'office2/Easy/P010/',
  'seasonsforest_winter/Hard/P010/',
  'westerndesert/Easy/P002/',
  'amusement/Easy/P007/',
  'gascola/Hard/P006/',
  'abandonedfactory_night/Easy/P005/',
  'westerndesert/Easy/P004/',
  'office2/Easy/P000/',
  'gascola/Hard/P002/',
  'abandonedfactory_night/Easy/P001/',
  'amusement/Hard/P004/'],
 ['office2/Hard/P004/',
  'hospital/Hard/P039/',
  'endofworld/Hard/P006/',
  'hospital/Easy/P018/',
  'oldtown/Hard/P003/',
  'abandonedfactory/Easy/P001/',
  'abandonedfactory/Hard/P002/',
  'seasonsforest/Easy/P003/',
  'office2/Hard/P000/',
  'gascola/Easy/P008/'],
 ['japanesealley/Hard/P003/',
  'seasonsforest/Easy/P005/',
  'seasonsforest_winter/Hard/P018/',
  'amusement/Hard/P002/',
  'office/Hard/P000/',
  'gascola/Easy/P005/',
  'hospital/Easy/P001/',
  'hospital/Easy/P020/',
  'soulcity/Hard/P005/',
  'neighborhood/Hard/P005/'])





class DeepVONet(nn.Module):
    def __init__(self):
        super(DeepVONet, self).__init__()
        w,h = image_size
        k = 7; s=2; p=3; fo=6; fn=64
        self.conv1 = nn.Sequential(  
            nn.Conv2d(fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.BatchNorm2d(fn), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.2)
        )
        w,h = int((w - k + 2 * p) / s) + 1, int((h-k+2*p) / s) + 1
#         print(w,h)
        k = 5; s=2; p=2; fo = fn; fn = 128
        self.conv2 = nn.Sequential( 
            nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.BatchNorm2d(fn),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.2)
        )
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
#         print(w,h)

        k = 5; s=2; p=2; fo =fn;fn = 256
        self.conv3 = nn.Sequential(
            nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.BatchNorm2d(fn),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.2)
        )
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
#         print(w,h)

        k = 3; s=1; p=1; fo = fn; fn = 256
        self.conv3_1 = nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p))
        self.relu3_1 = nn.LeakyReLU(0.1, inplace=True)
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
        k = 3; s=2; p=1; fo = fn; fn = 512
        self.conv4 = nn.Sequential(
            nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.BatchNorm2d(fn),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.2)
        )
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
#         print(w,h)

        k = 3; s=1; p=1; fo = fn; fn = 512
        self.conv4_1 = nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p))
        self.relu4_1 = nn.LeakyReLU(0.1, inplace=True)
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
        k = 3; s=2; p=1; fo = fn; fn = 512
        self.conv5 = nn.Sequential(
            nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.BatchNorm2d(fn),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.2)
        )
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
#         print(w,h)

        k = 3; s=1; p=1; fo = fn; fn = 512
        self.conv5_1 = nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p))
        self.relu5_1 = nn.LeakyReLU(0.1, inplace=True)
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
        k = 3; s=2; p=1; fo = fn; fn = 1024
        self.conv6 = nn.Sequential( 
            nn.Conv2d (fo, fn, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.BatchNorm2d(fn),
            nn.Dropout2d(p=0.2)
        )
        w,h = int((w-k+2*p) / s) + 1, int((h-k+2*p) / s) + 1
#         print(w,h)

        self.lstm1 = nn.LSTMCell(w*h*1024, 100)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTMCell(100, 100)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=100, out_features=7) #changed out to 7
        self.final_wh = (w,h)
        self.reset_hidden_states()

    def reset_hidden_states(self, size=1, zero=True):
        if zero == True:
            self.hx1 = Variable(torch.zeros(size, 100))
            self.cx1 = Variable(torch.zeros(size, 100))
            self.hx2 = Variable(torch.zeros(size, 100))
            self.cx2 = Variable(torch.zeros(size, 100))
        else:
            self.hx1 = Variable(self.hx1.data)
            self.cx1 = Variable(self.cx1.data)
            self.hx2 = Variable(self.hx2.data)
            self.cx2 = Variable(self.cx2.data)

        if next(self.parameters()).is_cuda == True:
            self.hx1 = self.hx1.cuda()
            self.cx1 = self.cx1.cuda()
            self.hx2 = self.hx2.cuda()
            self.cx2 = self.cx2.cuda()
        if USE_TPU == True:
            self.hx1 = self.hx1.to(device_tpu)
            self.cx1 = self.cx1.to(device_tpu)
            self.hx2 = self.hx2.to(device_tpu)
            self.cx2 = self.cx2.to(device_tpu)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu1(x)
        x = self.conv2(x)
        # x = self.relu2(x)
        x = self.conv3(x)
        # x = self.relu3(x)
#         x = self.conv3_1(x)
#         x = self.relu3_1(x)
        x = self.conv4(x)
        # x = self.relu4(x)
#         x = self.conv4_1(x)
#         x = self.relu4_1(x)
        x = self.conv5(x)
        # x = self.relu5(x)
#         x = self.conv5_1(x)
#         x = self.relu5_1(x)
        x = self.conv6(x)
        x = x.view(x.size(0), self.final_wh[0] * self.final_wh[1] * 1024)
        self.hx1, self.cx1 = self.lstm1(x, (self.hx1, self.cx1))
        x = self.dropout1(self.hx1)
        self.hx2, self.cx2 = self.lstm2(x, (self.hx2, self.cx2))
        x = self.dropout2(self.hx2)
        x = self.fc(x)
        return x


def default_image_loader(path):
    if isinstance(path, str):
#         print("loading from local")
        return Image.open(path).convert('RGB') #.transpose(0, 2, 1)
    return Image.fromarray(path)

class VisualOdometryDataLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, trajectory_length=10, transform=None, test=False,
                 loader=default_image_loader):
        self.base_path = datapath
        if isinstance(datapath[0], list):
            self.mixed_batch = True
            self.batch_size = len(datapath)

        else:
            self.mixed_batch = False
            self.batch_size = 1

        self.images=[[None for j in range(trajectory_length+1)] for i in range(self.batch_size)]
        self.how_far_have_we_gotten = [-1 for i in range(self.batch_size)]
        self.individual_lengths = [len(datapath[i]) for i in range(self.batch_size)]

        if test:
            self.sequences = ['01']
        else:
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            # self.sequences = ['01']

        # self.timestamps = self.load_timestamps()
        self.size = 0
        self.sizes = []
        self.poses = self.load_poses()
        self.trajectory_length = trajectory_length

        self.transform = transform
        self.loader = loader
        print(self.sizes, self.size, self.__len__())
    
    def load_poses(self):
        all_poses = []
        if not self.mixed_batch:
            # for sequence in self.sequences:
            f = read_text_file("/".join(self.base_path[0].split("/")[:3]) + (lambda x: "/pose_left" if "left" in x else "/pose_right")(self.base_path[0]) + '.txt')
            poses = np.array([[float(x) for x in line.split(" ")] for line in f.strip().split("\n")], dtype=np.float32)
      #         all_poses.append(poses)
            self.size = self.size + len(poses)
            self.sizes.append(len(poses))
            return poses
        else:
            for i, path in enumerate(self.base_path):
#                 f = read_text_file("/".join(path[0].split("/")[:3]) + (lambda x: "/pose_left" if "left" in x else "/pose_right")(path[0]) + '.txt')
#                 poses = np.array([[float(x) for x in line.split(" ")] for line in f.strip().split("\n")], dtype=np.float32)
#                 print(len(self.images[i]))
                poses = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for line in range(self.individual_lengths[i])], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
            return all_poses
    """ 
    def load_timestamps(self, sequence_path):
        for sequence in self.sequences:
            timestamp_file = os.path.join(self.sequence_path, 'times.txt')

            # Read and parse the timestamps
            timestamps = []
            with open(timestamp_file, 'r') as f:
                for line in f.readlines():
                    t = datetime.timedelta(seconds=float(line))
                    timestamps.append(t)
            return timestamps
    """

    def get_image(self, index, path):
        # image_path = os.path.join(self.base_path, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
#         image_path = os.path.join(self.base_path, '%06d' % index + '_left.png')
#         print(path[index])
        # if mixed_batch == True:
        #     image_path = read_image_file(self.base_path[(index % self.trajectory_length) % self.batch_size][int(index/self.batch_size)])
        # else:
#         print(path[index])
        image_path = None if not os.path.exists(path[index]) else path[index]
#         print("If not none then should be string: ", image_path)
        while image_path is None:
            try:
                image_path = read_image_file(path[index])
            except:
                print("trying again")

        image = self.loader(image_path)
        return image

    def __getitem__(self, index):
        # print("the index in __getitem__ is: ", index)
        sequence = 0
        sequence_size = 0
        # for size in self.sizes:
        #     if index < size-self.trajectory_length:
        #         sequence_size = size
        #         break
        #     index = index - (size-self.trajectory_length)
        #     sequence = sequence + 1
        
        # if (sequence >= len(self.sequences)):
        #     sequence = 0
        images_stacked = []
        traj_num = index % self.batch_size
        start_index = int(index / self.batch_size)
        path = self.base_path
        odometries = []
#         print("Requested index: ", start_index, " How far we've gotten: ", self.how_far_have_we_gotten[traj_num])
        if self.mixed_batch:
            path = self.base_path[traj_num]
        
        for i in range(start_index, start_index+self.trajectory_length):
            if self.how_far_have_we_gotten[traj_num] < i:
                # print("#need to load 2 images")
                # print("ssaving said images at loc ", (i % self.trajectory_length) ,((i+1) % self.trajectory_length) )
                img1 = self.get_image(i, path)
                img2 = self.get_image(i+1, path)
                self.how_far_have_we_gotten[traj_num] = i+1
                if self.transform is not None:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)
                self.images[traj_num][i % (self.trajectory_length + 1)] = img1
                self.images[traj_num][(i + 1) % (self.trajectory_length + 1)] = img2
                # del img1
                # del img2
            elif self.how_far_have_we_gotten[traj_num] < i+1:
                # print("#need to load 1 image")
                # print("ssaving said image at loc ", ((i+1) % self.trajectory_length) )
                img1 = self.get_image(i+1, path)
                self.how_far_have_we_gotten[traj_num] = i+1
                if self.transform is not None:
                    img1 = self.transform(img1)
                self.images[traj_num][(i + 1) % (self.trajectory_length + 1)] = img1
                # del img1
            pose1 = self.get6DoFPose(self.poses[traj_num][i])
            pose2 = self.get6DoFPose(self.poses[traj_num][i+1])
            odom = pose2 - pose1
            images_stacked.append(np.concatenate([self.images[traj_num][i % (self.trajectory_length + 1)], self.images[traj_num][(i+1) % (self.trajectory_length + 1)]], axis=0))
            odometries.append(odom)
#         print(index, images_stacked, odometries)
        return np.asarray(images_stacked), np.asarray(odometries)

    def __len__(self):
#         return self.size - (self.trajectory_length * len(self.sequences))
        return (min(self.sizes) - self.trajectory_length)*self.batch_size 
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        assert(self.isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if  not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z], dtype=np.float32)

    def get6DoFPose(self, p):
        # pos = np.array([p[3], p[7], p[11]])
        # R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
        # angles = self.rotationMatrixToEulerAngles(R)
        pos = np.array([p[0], p[1], p[2]]) #Cartesian coordinates
        angles = np.array([p[3], p[4], p[5], p[6]]) #Quaternions
        return np.concatenate((pos, angles))

# if __name__ == "__main__":
#     db = VisualOdometryDataLoader("/data/KITTI/dataset/")
#     img1, img2, odom = db[1]
#     print (odom)

#     import matplotlib.pyplot as plt

#     f, axarr = plt.subplots(2,2)
#     axarr[0,0].imshow(img1)
#     axarr[0,1].imshow(img2)
#     plt.show()

    
USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
K = 100.

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)

def train_model(model, train_loader, criterion, optimizer, epoch, batch_size, trajectory_length, test_loader):
    # switch to train mode
    losses = []
    valid_losses = []
    log_frequency = 50
    for batch_idx, (images_stacked, odometries_stacked) in tqdm(enumerate(train_loader)):
        # print(images_stacked.shape, odometries_stacked.shape)
        if USE_CUDA:
            images_stacked, odometries_stacked = images_stacked.cuda(), odometries_stacked.cuda()
        if USE_TPU:
            images_stacked, odometries_stacked = images_stacked.to(device_tpu), odometries_stacked.to(device_tpu)

        images_stacked = images_stacked.permute(1, 0, 2, 3, 4)
        images_stacked, odometries_stacked = Variable(images_stacked), Variable(odometries_stacked)

        estimated_odometries = Variable(torch.zeros(odometries_stacked.shape))
        estimated_odometries = estimated_odometries.permute(1, 0, 2)
        if USE_CUDA:
            estimated_odometries = estimated_odometries.cuda()
        if USE_TPU:
            estimated_odometries = estimated_odometries.to(device_tpu)

#         print(images_stacked.shape)
        model.reset_hidden_states(size=batch_size, zero=True)
        for t in range(trajectory_length):
            # compute output
            estimated_odometry = model(images_stacked[t])
            estimated_odometries[t] = estimated_odometry
            
        estimated_odometries = estimated_odometries.permute(1, 0, 2)
            
        loss = 0.1 * K * criterion(estimated_odometries[:,:,:3], odometries_stacked[:,:,:3]) + K * criterion(estimated_odometries[:,:,3:], odometries_stacked[:,:,3:])

        # compute gradient and do optimizer step
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        loss.backward()
        if USE_TPU:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
        
#         if losses:
#             if loss.data.cpu().numpy()/(trajectory_length*batch_size) > 1.5*losses[-1]:

#                 for t in range(trajectory_length):
#                     # compute output
#                     f, axarr = plt.subplots(1,2, figsize=(25,25))
#                     axarr[0].imshow(unnormalize(images_stacked[t][0][:3]).data.cpu().numpy().transpose((1, 2, 0)))
#                     axarr[1].imshow(unnormalize(images_stacked[t][0][3:]).data.cpu().numpy().transpose((1, 2, 0)))
#                     plt.show()
                # print (epoch, batch_idx, loss.data.cpu()[0])
        if batch_idx % log_frequency == 0 and len(losses) > 0:
            val_loss = test(model, val_list, trajectory_length=10, validation_steps=10, preprocess=preprocess, test_loader=test_loader)
            valid_losses.append(val_loss)
            if len(valid_losses)>1:
                print("Observe if the numbers match: ", log_frequency*len(valid_losses), len(losses), batch_idx)
                plt.plot(range(log_frequency, (log_frequency)*(len(valid_losses)+1), log_frequency), valid_losses)
                plt.title("Validation loss batch")                
                plt.plot(range(len(losses)), losses)
#                 plt.show()
                plt.savefig('./training_progress/validation_loss_{}.png'.format(datetime.datetime.now()))
#                 plt.show()
                plt.close()
            model.train()
            model.training = True
            gc.collect()
        losses.append(loss.data.cpu().numpy()/(trajectory_length*batch_size))
            
#                 plt.imshow(images_stacked[0][0].data.cpu().numpy().transpose(1, 2, 0))
#                 plt.show()
            # f, axarr = plt.subplots(1,2)
            # axarr[0].imshow(unnormalize(images_stacked[t][0][:3]).data.cpu().numpy().transpose((1, 2, 0)))
            # axarr[1].imshow(unnormalize(images_stacked[t][0][3:]).data.cpu().numpy().transpose((1, 2, 0)))
            # plt.show()
    #         print(estimated_odometries, odometries_stacked)
            # print(pd.DataFrame(np.reshape(estimated_odometries[0][t].cpu().detach().numpy(), (-1, 7))))
            # print(pd.DataFrame(np.reshape(odometries_stacked[0][t].cpu().detach().numpy(), (-1, 7))))
        # print (epoch, batch_idx, loss.data.cpu().numpy()/(trajectory_length*batch_size))
    return (np.sum(losses)/batch_idx)
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def train(model, datapath, checkpoint_path, epochs, trajectory_length, args):
    model.train()
    model.training = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {} #check again

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses_epoch = []

    val_batch_size = len(val_list)
    val_traj_len = 10
    val_datapath = [get_image_list(trajectory, left_right="left") for trajectory in val_list]

    test_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(val_datapath, 
                                                                       trajectory_length=val_traj_len, 
                                                                       transform=preprocess, 
                                                                       test=True), 
                                              batch_size=val_batch_size, 
                                              shuffle=False, **kwargs)
    val_iter = 10

    for epoch in range(1, epochs + 1):
        losses_traj = []
        val_loss = []
        for trajectory_batch in batch(datapath,args.bsize):
            print("current trajectory is: ", trajectory_batch)
            datapath1 = [get_image_list(trajectory, left_right="left") for trajectory in trajectory_batch]
            train_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath1, 
                                                                                trajectory_length=trajectory_length, 
                                                                                transform=preprocess), 
                                                       batch_size=args.bsize, 
                                                       shuffle=False, 
                                                       drop_last=True, 
                                                       **kwargs)

            # train for one epoch
            lss = train_model(model, train_loader, criterion, optimizer, epoch, args.bsize, trajectory_length, test_loader)
    #        # evaluate on validation set
          #  acc = test(test_loader, tripletnet, criterion, epoch)
    #
    #        # remember best acc and save checkpoint
    #        is_best = acc > best_acc
    #        best_acc = max(acc, best_acc)
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }

            val_loss.append(test(model, val_list, val_traj_len, val_iter, preprocess, test_loader))

            losses_traj.append(lss)

            if len(losses_traj) > 1:
                plt.plot(range(len(losses_traj)), losses_traj)
                plt.plot(range(len(val_loss)), val_loss)
                plt.legend(["train", "valid"])
#                 plt.show()
                plt.savefig("training_progress/train_valid_curve_{}.png".format(".".join([ trajectory_batch[t] for t in (0,-1)]).replace("/", ".")))
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_{0}_{1}.pth".format(epoch, ".".join([ trajectory_batch[t] for t in (0,-1)]).replace("/", "."))))
            # del train_loader
        losses_epoch.append(np.sum(losses_traj))
        plt.plot(range(len(losses_epoch)), losses_epoch)
        plt.show()
      

def test(model, datapath, trajectory_length, validation_steps, preprocess, test_loader):
    model.eval()
    model.training = False
    batch_size = len(datapath)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # datapath1 = get_image_list(trajectory, left_right="left") 
    # print("current trajectory is: ", datapath)
    
    criterion = torch.nn.MSELoss()
    losses = []
    true_len = 0

    model.reset_hidden_states(size=batch_size, zero=True)
    for batch_idx, (images_stacked, odometries_stacked) in tqdm(enumerate(test_loader)):
        if batch_idx % trajectory_length:
#             print("skips execution at batch_idx: ", batch_idx)
            continue
        if USE_CUDA:
            images_stacked, odometries_stacked = images_stacked.cuda(), odometries_stacked.cuda()
        if USE_TPU:
            images_stacked, odometries_stacked = images_stacked.to(device_tpu), odometries_stacked.to(device_tpu)
        images_stacked = images_stacked.permute(1, 0, 2, 3, 4)
        images_stacked, odometries_stacked = Variable(images_stacked), Variable(odometries_stacked)

        estimated_odometries = Variable(torch.zeros(odometries_stacked.shape))
        estimated_odometries = estimated_odometries.permute(1, 0, 2)
        if USE_CUDA:
            estimated_odometries = estimated_odometries.cuda()
        if USE_TPU:
            estimated_odometries = estimated_odometries.to(device_tpu)

        model.reset_hidden_states(size=batch_size, zero=False)
        for t in range(trajectory_length):
            estimated_odometry = model(images_stacked[t])
            estimated_odometries[t] = estimated_odometry

        estimated_odometries = estimated_odometries.permute(1, 0, 2)

        loss = 0.1 * K * criterion(estimated_odometries[:,:,:3], odometries_stacked[:,:,:3]) + K * criterion(estimated_odometries[:,:,3:], odometries_stacked[:,:,3:])
        losses.append(loss.data.cpu().numpy())
        true_len += trajectory_length
        
#         for t in range(trajectory_length):
#             # compute output
#             f, axarr = plt.subplots(1,2, figsize=(25,25))
#             axarr[0].imshow(unnormalize(images_stacked[t][0][:3]).data.cpu().numpy().transpose((1, 2, 0)))
#             axarr[1].imshow(unnormalize(images_stacked[t][0][3:]).data.cpu().numpy().transpose((1, 2, 0)))
#             plt.show()

        
        if true_len >= validation_steps:
            print("breaks at batch_idx: ", batch_idx)
            break
        
    return np.sum(losses)/(true_len*batch_size)

    """
    with open(os.path.join(datapath, "index.txt"), 'r') as reader:
        for index in reader:
            index = index.strip()
            images_path = []
            with open(os.path.join(datapath, index, "index.txt"), 'r') as image_reader:
                for image_path in image_reader:
                    images_path.append(image_path.strip())

            model.reset_hidden_states(size=1, zero=True)
            for image_index in range(len(images_path)-1):
                model.reset_hidden_states(size=1, zero=False)
                image1 = Image.open(os.path.join(datapath, index, images_path[image_index])).convert('RGB')
                image2 = Image.open(os.path.join(datapath, index, images_path[image_index+1])).convert('RGB')
                image1_tensor = preprocess(image1)
                image2_tensor = preprocess(image2)

                # plt.figure()
                # plt.imshow(images_stacked.cpu().numpy().transpose((1, 2, 0)))
                # plt.show()

                images_stacked = torch.from_numpy(np.concatenate([image1_tensor, image2_tensor], axis=0))
                images_stacked.unsqueeze_(0)
                images_stacked = Variable(images_stacked).cuda()
                odom = model(images_stacked)
                print (image_index, image_index+1, odom.data.cpu())
                del images_stacked, odom, image1_tensor, image2_tensor
    """
    
normalize = transforms.Normalize(
    #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
    mean=[127. / 255., 127. / 255., 127. / 255.],
    std=[1 / 255., 1 / 255., 1 / 255.]
#     mean=[0., 0., 0.],
#     std=[1, 1, 1]
    
)

preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize
])

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


unnormalize = UnNormalize(mean=(127. / 255., 127. / 255., 127. / 255.),
    std=(1 / 255., 1 / 255., 1 / 255.))

