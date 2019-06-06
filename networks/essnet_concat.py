import torch
import torch.nn as nn
from networks.base.modules import ConcatFeatRegression
from networks.base.basenet import BaseNet
from networks.base.resnet import ResNet34
                  
    
class EssNetConcat(BaseNet):
    def __init__(self, config):
        print('Build up EssNetConcat model...')
        super().__init__(config)
        self.extract = ResNet34()
        self.regress = ConcatFeatRegression(target='ess')
        self.to(self.device)
        self.init_weights_(weights_dict=config.weights_dict, pretrained=True)
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
        feat2 = self.extract(x2)
        ess = self.regress(feat1, feat2)
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
    
    def init_weights_(self, weights_dict=None, pretrained=True):
        if weights_dict:
            print('Load all model parameters from weights dict')
            self.load_state_dict(weights_dict)
        elif pretrained:
            self.apply(self.kaiming_normal_init_func_)            
            pretrained_weights = self.extract.get_pretrained_weights_(prefix='extract')
            self.load_state_dict(pretrained_weights, strict=False)
            print('Load pretrained Resnet34 for feature extraction part and Kaiming init the left')            
        else:
            print('Kaiming Initialize all model parameters')
            self.apply(self.kaiming_normal_init_func_)
    
    def loss_(self, batch):
        x1, x2, label = self.get_inputs_(batch, with_label=True)
        criterion = nn.MSELoss()
        pred = self.forward(x1, x2)
        loss = criterion(pred, label)
        return loss