import torch
import torch.nn as nn

from networks.base.basenet import BaseNet
from networks.base.modules import MatchingFeatRegression
from networks.base.resnet import ResNet34
from networks.util.epipolar import epipolar_residual_error, essmats2fundmats

from networks.ncn.conv4d import Conv4d
from networks.ncn.model import featureL2Norm, MutualMatching, FeatureCorrelation, maxpool4d, NeighConsensus     
    
class NCEssNet(BaseNet):
    def __init__(self, config, ncn_weight=None, half_precision=False, early_feat=False, relocalization_k_size=0):
        print('Build up NCEssNet model...')
        super().__init__(config)
        self.early_feat = config.early_feat
        self.half_precision = half_precision
        self.early_feat = early_feat
        self.relocalization_k_size = relocalization_k_size  #
        
        self.extract = ResNet34()
        self.combine = FeatureCorrelation(shape='4D', normalization=False)
        self.ncn = NeighConsensus(kernel_sizes=[3, 3], channels=[16, 1])  # Needed for weight initialization
        self.regress = MatchingFeatRegression(in_size=config.feat_size, target='ess')
        self.to(self.device)
        self.init_weights_(weights_dict=config.weights_dict, ncn_weight=ncn_weight, pretrained=True)
        self.set_optimizer_(config)
        if config.training:
            self.loss_type = config.loss_type        
        if self.half_precision:
            for p in self.ncn.parameters():
                p.data=p.data.half()
            for l in self.ncn.conv:
                if isinstance(l,Conv4d):
                    l.use_half=True

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
        corr4d, _  = self.forward_corr4d(x1, x2)
        N, _, H, W, _, _ = corr4d.size()
        corr4d = corr4d.view(N, H, W, H*W).permute(0, 3, 1, 2) # Adapt to NCHW
        ess = self.regress(corr4d)
        if self.config.ess_proj:
            ess = self.projection_layer(ess)
        return ess
    
    def forward_corr4d(self, x1, x2):
        feat1 = self.extract(x1, early_feat=self.early_feat)        
        feat2 = self.extract(x2, early_feat=self.early_feat)     # Shared weights
        
        # feature normalization
        feat1 = featureL2Norm(feat1)
        feat2 = featureL2Norm(feat2)
        
        if self.half_precision:
            feat1=feat1.half()
            feat2=feat2.half()
        
        # feature correlation
        corr4d = self.combine(feat1, feat2)

        # do 4d maxpooling for relocalization
        if self.relocalization_k_size>1:
            corr4d,max_i,max_j,max_k,max_l=maxpool4d(corr4d, k_size=self.relocalization_k_size)
            
        corr4d = MutualMatching(corr4d)
        corr4d = self.ncn(corr4d)
        corr4d = MutualMatching(corr4d)
        if self.relocalization_k_size>1:
            delta4d=(max_i,max_j,max_k,max_l)
            return (corr4d,delta4d)
        else:
            return (corr4d, None)
    
    def get_inputs_(self, batch, with_label=True):
        im1, im2 = batch['im_pairs']
        im1, im2 = im1.to(self.device), im2.to(self.device)
        if with_label:
            label = batch['ess_vec'].to(self.device)
            return im1, im2, label
        else:
            return im1, im2
    
    def init_weights_(self, weights_dict=None, ncn_weight=None, pretrained=True):
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
        if ncn_weight:
            print('Overwrite NCN module...')
            self.ncn.load_state_dict(ncn_weight)

    def epipolar_loss(self, batch):
        im1, im2 = batch['im_pairs']
        im1, im2 = im1.to(self.device), im2.to(self.device)
        ess_vecs = self.forward(im1, im2)        
        ess_mats = ess_vecs.view(-1, 3, 3)
        _, _, K1_inv, K2_inv = batch['intrinsics']
        K1_inv = K1_inv.to(self.device)
        K2_inv = K2_inv.to(self.device)   
        F = essmats2fundmats(K1_inv, K2_inv, ess_mats)
        pts1, pts2 = batch['virtual_pts']
        pts1 = pts1.to(self.device)
        pts2 = pts2.to(self.device)  
        loss = epipolar_residual_error(pts1, pts2, F)
        loss = loss.mean()
        return loss    
    
    def loss_(self, batch):
        if 'epipolar' in self.loss_type:
            loss = self.epipolar_loss(batch)
        x1, x2, label = self.get_inputs_(batch, with_label=True)
        pred = self.forward(x1, x2)
        if 'signed_mse' in self.loss_type:
            criterion = nn.MSELoss(reduce=False)
            loss_pos = criterion(pred, label)
            loss_neg = criterion(-pred, label)
            loss = torch.mean(torch.min(loss_pos, loss_neg))
        else:
            criterion = nn.MSELoss()
            loss = criterion(pred, label)
        return loss