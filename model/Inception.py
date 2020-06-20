import torch
import torch.nn as nn


##Inception v1
class Inceptionv1Module(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2_1, out_channels2_2, out_channels3_1, out_channels3_2, out_channels4):
        super(Inceptionv1Module, self).__init__()
        
        self.conv1branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels1),
                nn.ReLU(inplace=True),
                )
        self.conv3branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels2_1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels2_1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels2_1, out_channels2_2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels2_2),
                nn.ReLU(inplace=True),
                )
        self.conv5branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels3_1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels3_1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels3_1, out_channels3_2, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(out_channels3_2),
                nn.ReLU(inplace=True),
                )
        self.poolbranch = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels4, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels4),
                nn.ReLU(inplace=True),
                )
        
    def forward(self, x):
        out1 = self.conv1branch(x)
        out2 = self.conv3branch(x)
        out3 = self.conv5branch(x)
        out4 = self.poolbranch(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out
    
class Inceptionv1OutModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inceptionv1OutModule, self).__init__()
        self.convlayer = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),
                nn.Conv2d(in_channels, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                )
        self.fclayer1 = nn.Sequential(
                nn.Linear(128*4*4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.7),
                )
        self.fclayer2 = nn.Linear(1024, out_channels)
        
    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(x.size(0), -1)
        x = self.fclayer1(x)
        out = self.fclayer2(x)
        return out
    
class Inceptionv1(nn.Module):
    def __init__(self, num_classes=1000, is_train=True):
        super(Inceptionv1, self).__init__()
        self.is_train = is_train
        self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        self.block2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        self.block3 = nn.Sequential(
                Inceptionv1Module(in_channels=192, out_channels1=64, out_channels2_1=96, out_channels2_2=128, out_channels3_1=16, out_channels3_2=32, out_channels4=32),
                Inceptionv1Module(in_channels=256, out_channels1=128, out_channels2_1=128, out_channels2_2=192, out_channels3_1=32, out_channels3_2=96, out_channels4=64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        self.block4_1 = nn.Sequential(
                Inceptionv1Module(in_channels=480, out_channels1=192, out_channels2_1=96, out_channels2_2=208, out_channels3_1=18, out_channels3_2=48, out_channels4=64),
                )
        self.out_block1 = Inceptionv1OutModule(512, num_classes)
        self.block4_2 = nn.Sequential(
                Inceptionv1Module(in_channels=512, out_channels1=160, out_channels2_1=112, out_channels2_2=224, out_channels3_1=24, out_channels3_2=64, out_channels4=64),
                Inceptionv1Module(in_channels=512, out_channels1=128, out_channels2_1=128, out_channels2_2=256, out_channels3_1=24, out_channels3_2=64, out_channels4=64),
                Inceptionv1Module(in_channels=512, out_channels1=112, out_channels2_1=144, out_channels2_2=288, out_channels3_1=32, out_channels3_2=64, out_channels4=64),
                )
        self.out_block2 = Inceptionv1OutModule(528, num_classes)
        self.block4_3 = nn.Sequential(
                Inceptionv1Module(in_channels=528, out_channels1=256, out_channels2_1=160, out_channels2_2=320, out_channels3_1=32, out_channels3_2=128, out_channels4=128),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        self.block5 = nn.Sequential(
                Inceptionv1Module(in_channels=832, out_channels1=256, out_channels2_1=160, out_channels2_2=320, out_channels3_1=32, out_channels3_2=128, out_channels4=128),
                Inceptionv1Module(in_channels=832, out_channels1=384, out_channels2_1=192, out_channels2_2=384, out_channels3_1=48, out_channels3_2=128, out_channels4=128),
                )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out1 = x = self.block4_1(x)
        out2 = x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        if self.is_train:
            print(out1.shape)
            out1 = self.out_block1(out1)
            out2 = self.out_block2(out2)
            return out1, out2, out
        else:
            return out
        
        
if __name__ == '__main__':
    net = Inceptionv1(1000)
    print(net)
    
    input = torch.randn(1, 3, 224, 224)
    out1, out2, out = net(input)
    print(out1.shape)
    print(out2.shape)
    print(out.shape)