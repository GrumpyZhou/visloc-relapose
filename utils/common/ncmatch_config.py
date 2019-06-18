import torch
from utils.common.setup_helper import load_weights
# TODO: Add training config for NCMatchNet and its setup function?

class NCMatchEvalConfig:
    def __init__(self, weights_dir=None, feat_weights=None, ncn_weights=None, early_feat=False,
                 relocalization_k_size=2, fe_finetune_params=None, half_precision=False,
                 gpu=0, odir=None):
        # Fixed Part
        self.seed = 1
        self.num_workers = 0
        self.training = False
        self.batch_size = 16
        
        # Model setting
        self.fe_finetune_params = fe_finetune_params 
        self.relocalization_k_size = relocalization_k_size
        self.half_precision = half_precision
        self.early_feat = early_feat
        
        # Load model weights  
        self.device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
        self.weights_dict = load_weights(weights_dir, self.device)
        self.feat_weights = load_weights(feat_weights, self.device)
        self.ncn_weights = load_weights(ncn_weights, self.device)
        
        # Logging
        self.odir = odir