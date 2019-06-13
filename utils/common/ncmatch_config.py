import torch

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
        map_location = lambda storage, loc: storage.cuda(self.device.index) if torch.cuda.is_available() else storage
        self.weights_dict = torch.load(weights_dir, map_location=map_location) if weights_dir else None
        self.feat_weights = torch.load(feat_weights, map_location=map_location) if feat_weights else None
        self.ncn_weights = torch.load(ncn_weights, map_location=map_location) if ncn_weights else None
        
        # Logging
        self.odir = odir