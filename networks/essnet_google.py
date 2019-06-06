import torch
import torch.nn as nn
from networks.base.modules import Correlation, MatchingFeatRegression
from networks.base.basenet import BaseNet
from networks.base.googlenet import GoogLeNet
                 
class EssNetGL(BaseNet):
    def __init__(self, config):
        print('Build up EssNet model...')
        super().__init__(config)
        self.extract = GoogLeNet(with_aux=False)
        self.combine = Correlation()
        self.regress = MatchingFeatRegression(target='ess')
        self.to(self.device)
        self.init_weights_(config.weights_dict)
        self.set_optimizer_(config)
    
    def proj_to_ess_space(self, E):
        u,s,v = E.svd()
        vh = v.t()
        a = (s[0] + s[1]) / 2
        s_ = torch.Tensor([a, a, 0]).to(E.device)
        E = torch.matmul(torch.matmul(u, torch.diag(s_)), vh)
        return E
    
    def projection_layer(self, ess_preds):
        ess_preds_ = []
        for j, ess in enumerate(ess_preds):
            E = ess.view((3,3))
            E = self.proj_to_ess_space(E)
            ess_preds_.append(E.view(1,9))
        return torch.cat(ess_preds_, dim=0)
        
    def forward(self, x1, x2):
        feat1 = self.extract(x1)        
        feat2 = self.extract(x2) # Shared weights
        feat = self.combine(feat1, feat2)
        ess = self.regress(feat)
        if self.config.ess_proj:
            ess = self.projection_layer(ess)
        return ess
    
    def get_inputs_(self, batch, with_label=True):
        im1, im2 = batch['im_pairs']
        im1, im2 = im1.to(self.device), im2.to(self.device)
        if with_label:
            label = batch['ess_vec'].to(self.device)
            return im1, im2, label
        else:
            return im1, im2
    
    def init_weights_(self, weights_dict):
        if weights_dict is None:
            print('Xavier Initialize all model parameters')
            self.apply(self.xavier_init_func_)
        elif len(weights_dict.items()) == len(self.state_dict()):
            print('Load all model parameters from weights dict')
            self.load_state_dict(weights_dict)
        else:
            print('Load part of model parameters and Xavier init the left')
            self.apply(self.xavier_init_func_)
            self.load_state_dict(weights_dict, strict=False)    
    
    def loss_(self, batch):
        x1, x2, label = self.get_inputs_(batch, with_label=True)
        criterion = nn.MSELoss()
        pred = self.forward(x1, x2)
        loss = criterion(pred, label)
        return loss
