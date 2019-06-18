import torch
import torch.nn as nn

from networks.base.basenet import BaseNet
from networks.base.resnet import ResNet34

from networks.ncn.conv4d import Conv4d
from networks.ncn.model import featureL2Norm, MutualMatching, FeatureCorrelation, maxpool4d, NeighConsensus     
    
class NCMatchNet(BaseNet):
    def __init__(self, config):
        print('Build up NC-MatchNet model...')
        super().__init__(config)
        self.early_feat = config.early_feat
        self.half_precision = config.half_precision
        self.relocalization_k_size = config.relocalization_k_size  #
        print('Config: early_feat {} half_precision {} k_size {}'.format(self.early_feat, self.half_precision, 
                                                                 self.relocalization_k_size))
        self.extract = ResNet34()
        self.combine = FeatureCorrelation(shape='4D', normalization=False)
        self.ncn = NeighConsensus(kernel_sizes=[3, 3], channels=[16, 1])
        self.to(self.device)
        
        # Load pretrianed weights/models or update weights after loading
        # All weights are already loaded to the correct devices in advance
        self.init_weights_(weights_dict=config.weights_dict, pretrained=True) 
        if config.feat_weights is not None or config.ncn_weights is not None:
            self.update_weights_(config.feat_weights, config.ncn_weights)
        self.set_optimizer_(config)
        
        if self.half_precision:
            for p in self.ncn.parameters():
                p.data=p.data.half()
            for l in self.ncn.conv:
                if isinstance(l,Conv4d):
                    l.use_half=True
                    
    def forward_corr4d(self, x1, x2):
        return self.forward(x1, x2)
        
    def forward(self, x1, x2):
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
    
    def get_inputs_(self, batch):
        im_src = batch['source_image'].to(self.device)
        im_pos = batch['pos_target'].to(self.device)
        im_neg = batch['neg_target'].to(self.device)
        return im_src, im_pos, im_neg
    
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
    
    def update_weights_(self, feat_dict=None, ncn_dict=None):
        model_dict = self.state_dict()
        if feat_dict is not None:
            feat_dict = {k: v for k, v in feat_dict.items() if k in model_dict and k.startswith('extract')}
            print('Overwrite extractor weights, keys:{}'.format(len(feat_dict)))
            model_dict.update(feat_dict) 
        if ncn_dict is not None:
            print('Overwrite ncn weights...')
            ncn_dict = {k: v for k, v in ncn_dict.items() if k in model_dict and k.startswith('ncn')}
            model_dict.update(ncn_dict) 
        self.load_state_dict(model_dict)
        
    def cal_match_score(self, im_src, im_tar, normalize):
        if normalize is None:
            normalize = lambda x: x
        elif normalize == 'softmax':     
            normalize = lambda x: nn.functional.softmax(x, 1)
        elif normalize == 'l1':
            normalize = lambda x: x / (torch.sum(x, dim=1, keepdim=True) + 0.0001)
        
        # Mutual matching score
        corr4d, _ = self.forward(im_src, im_tar)
        batch_size = corr4d.size(0)
        feature_size = corr4d.size(2)
        nc_B_Avec=corr4d.view(batch_size, feature_size*feature_size, feature_size, feature_size)
        nc_A_Bvec=corr4d.view(batch_size, feature_size, feature_size, feature_size*feature_size).permute(0,3,1,2) # 
        nc_B_Avec = normalize(nc_B_Avec)
        nc_A_Bvec = normalize(nc_A_Bvec)
        scores_B,_= torch.max(nc_B_Avec, dim=1)
        scores_A,_= torch.max(nc_A_Bvec, dim=1)
        score = torch.mean(scores_A + scores_B) / 2
        return score

    def loss_(self, batch):
        im_src, im_pos, im_neg = self.get_inputs_(batch)
        score_pos = self.cal_match_score(im_src, im_pos, normalize='softmax')
        score_neg = self.cal_match_score(im_src, im_neg, normalize='softmax')
        loss = score_neg - score_pos  
        return loss
    
