import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from image_parameters import image_size
from torchvision import transforms
from torch.autograd import Variable


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

