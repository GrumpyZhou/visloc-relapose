import torch
import torch.nn as nn
from networks.base.basenet import BaseNet
from networks.base.resnet import ResNet34
            
class RelaPoseRegression(nn.Module):
    def __init__(self):
        super(RelaPoseRegression, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc_pose = nn.Sequential(nn.Linear(1024, 1024),
                                     nn.BatchNorm1d(1024), 
                                     nn.ReLU())
        self.fc_t = nn.Linear(1024, 3)
        self.fc_q = nn.Linear(1024, 4)

    def forward(self, x1, x2):
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)
        x = torch.cat([x1, x2], dim=1) 
        x = self.fc_pose(x.view(x.size(0), -1))
        t = self.fc_t(x)
        q = self.fc_q(x)
        return (t, q)

class RelaPoseNet(BaseNet):
    def __init__(self, config):
        print('Build up RelaPoseNet model...')
        super().__init__(config)
        self.extract = ResNet34()
        self.regress = RelaPoseRegression()
        if config.training:
            self.loss_type = config.loss_type
            if 'homo' in self.loss_type:  # Learned loss weighting during training
                sx, sq = config.homo_init
                # Variances variables to learn
                self.sx = nn.Parameter(torch.tensor(sx))
                self.sq = nn.Parameter(torch.tensor(sq))
            else:   # Fixed loss weighting with beta
                self.beta = config.beta # Default beta = 1

        self.to(self.device)
        self.init_weights_(weights_dict=config.weights_dict, pretrained=True)
        self.set_optimizer_(config)
            
    def forward(self, x1, x2):
        feat1 = self.extract(x1)        
        feat2 = self.extract(x2) # Shared weights
        pose = self.regress(feat1, feat2)
        return pose
    
    def get_inputs_(self, batch, with_label=True):
        im1, im2 = batch['im_pairs']
        im1, im2 = im1.to(self.device), im2.to(self.device)
        if with_label:
            t_lbl = batch['relv_t'].to(self.device)
            q_lbl = batch['relv_q'].to(self.device)
            return im1, im2, t_lbl, q_lbl
        else:
            return im1, im2

    def init_weights_(self, weights_dict=None, pretrained=True):
        if weights_dict:
            if len(weights_dict.items()) == len(self.state_dict()):
                print('Load all model parameters from weights dict')
                self.load_state_dict(weights_dict)
            else:
                print('Load part of model parameters and Kaiming init the left')
                self.apply(self.kaiming_normal_init_func_)
                self.load_state_dict(weights_dict, strict=False)
        elif pretrained:
            self.apply(self.kaiming_normal_init_func_)            
            pretrained_weights = self.extract.get_pretrained_weights_(prefix='extract')
            self.load_state_dict(pretrained_weights, strict=False)
            print('Load pretrained Resnet34 for feature extraction part and Kaiming init the left')            
        else:
            print('Kaiming Initialize all model parameters')
            self.apply(self.kaiming_normal_init_func_)
    
    def loss_(self, batch):
        x1, x2, t_lbl, q_lbl = self.get_inputs_(batch, with_label=True)
        criterion = nn.MSELoss()
        t_pred, q_pred = self.forward(x1, x2)
        loss_t = criterion(t_pred, t_lbl) 
        loss_q = criterion(q_pred, q_lbl)
        if 'homo' in self.loss_type:
            loss_func = lambda loss_t, loss_q: self.learned_weighting_loss(loss_t, loss_q, self.sx, self.sq)
        else:
            loss_func = lambda loss_t, loss_q: self.fixed_weighting_loss(loss_t, loss_q, beta=self.beta)
        loss = loss_func(loss_t, loss_q)
        #losses = [loss_t, loss_q]
        #if 'homo' in self.loss_type:
        #    losses += [self.sx, self.sq]
        return loss