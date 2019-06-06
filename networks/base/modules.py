import torch
import torch.nn as nn
import torch.nn.functional as F

class Correlation(nn.Module):
    def __init__(self):
        super(Correlation, self).__init__()
        
    def forward(self, x1, x2):
        (N,C,H,W) = x1.size()
        x1 = x1.view(N, C, H*W)
        x2 = x2.view(N, C, H*W)
        x = torch.bmm(x1.transpose(1, 2), x2)
        x = x.view(N, H, W, H*W).permute(0, 3, 1, 2) # Adapt to NCHW
        x = F.normalize(x, p=2, dim=1)
        return x

class MatchingFeatRegression(nn.Module):
    """This is a regression block for feature output of a correlation/matching layer
    The output feature of the matching layer is supposed to be 7x7x49
    The regression targets can be an essential matrix, a relative pose 
    or 5 angles representing a relative pose. The target is specified by
    setting 'target' to one of  ['ess', 'relapose', 'poseang'].
    """
    
    def __init__(self, in_size=7, target='ess'):
        super(MatchingFeatRegression, self).__init__()
        self.target = target
        self.conv1 = nn.Sequential(nn.Conv2d(in_size**2, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU()) # Input: 7x7x49
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU()) # Output: 5x5x64
        out_size = in_size - 3 +1 
        linear_size = out_size ** 2 * 64
        print('MatchingFeatRegression: in: {} out: {} linear: {}'.format(in_size, out_size, linear_size))
        if target == 'ess':
            self.fc = nn.Linear(linear_size, 9)
        elif target == 'relapose':
            self.fc_t = nn.Linear(linear_size, 3) # 5x5x64
            self.fc_q = nn.Linear(linear_size, 4)
        elif target == 'poseang':
            self.fc = nn.Linear(linear_size, 5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        if self.target == 'relapose':
            t = self.fc_t(x)
            q = self.fc_q(x)
            target = (t, q)
        else:
            target = self.fc(x)
        return target
    
class ConcatFeatRegression(nn.Module):
    def __init__(self, target='relapose'):
        super(ConcatFeatRegression, self).__init__()
        self.target = target
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1) # Input 7x7x512
        self.fc_pose = nn.Sequential(nn.Linear(1024, 1024),
                                     nn.BatchNorm1d(1024), 
                                     nn.ReLU())
        if target == 'ess':
            self.fc = nn.Linear(1024, 9)
        elif target == 'relapose':
            self.fc_t = nn.Linear(1024,  3) # 5x5x64
            self.fc_q = nn.Linear(1024, 4)
        elif target == 'poseang':
            self.fc = nn.Linear(1024, 5)
        
    def forward(self, x1, x2):
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)
        x = torch.cat([x1, x2], dim=1) 
        x = self.fc_pose(x.view(x.size(0), -1))
        if self.target == 'relapose':
            t = self.fc_t(x)
            q = self.fc_q(x)
            target = (t, q)
        else:
            target = self.fc(x)
        return target
