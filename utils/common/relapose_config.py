import argparse
import os
import torch
from utils.datasets.preprocess import get_pair_transform_ops
from utils.common.setup_helper import make_deterministic, config2str

def setup_config(config):
    print('Setup configurations...')
    make_deterministic(config.seed)   # Seeding

    # Setup logging dirs
    if not os.path.exists(config.odir):
        os.makedirs(config.odir)
    config.log = os.path.join(config.odir, 'log.txt') \
                 if config.training else os.path.join(config.odir, 'test_results.txt')
    config.ckpt_dir = os.path.join(config.odir, 'ckpt')
    if not os.path.exists(config.ckpt_dir) and config.training:
        os.makedirs(config.ckpt_dir)

    # Setup running devices
    config.device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    map_location = lambda storage, loc: storage.cuda(config.device.index) if torch.cuda.is_available() else storage
    print('Use device:{}.'.format(config.device))
    delattr(config, 'gpu')

    # Setup datasets    
    config.train_lbl_txt = config.abs_label_txt[0]
    config.test_lbl_txt = config.abs_label_txt[1]
    delattr(config, 'abs_label_txt') 

    # Define image preprocessing
    crop_type = 'random' if config.training else 'center'
    crop_size = config.crop
    # The full resent34 downscale input by 32
    config.feat_size = crop_size // 32 if not config.early_feat else crop_size // 16 
    config.ops = get_pair_transform_ops(config.rescale, crop=crop_type, 
                                        crop_size=crop_size, normalize=config.normalize)
    if config.training:
        config.val_ops = get_pair_transform_ops(config.rescale, crop='center',
                                                crop_size=crop_size, normalize=config.normalize)
    delattr(config, 'rescale')
    delattr(config, 'crop')
    delattr(config, 'normalize')

    # Model initialization 
    config.start_epoch = 0
    config.weights_dir = None
    config.weights_dict = None
    config.optimizer_dict = None
    if config.pretrained:
        config.weights_dir = config.pretrained
        config.weights_dict = torch.load(config.weights_dir, map_location=map_location)
    if config.resume and os.path.isfile(config.resume):
        config.weights_dir = config.resume
        checkpoint = torch.load(config.weights_dir, map_location=map_location)
        config.start_epoch = checkpoint['last_epoch'] + 1
        config.weights_dict = checkpoint['state_dict']
        config.optimizer_dict = checkpoint['optimizer']
    delattr(config, 'resume')
    delattr(config, 'pretrained')

    if config.training:
        # Setup optimizer
        optim_tag = 'lr{}_wd{}'.format(config.lr_init, config.weight_decay)
        config.lr_decay_factor = float(config.lr_decay[0])
        config.lrd_decay_step = int(config.lr_decay[1])
        if config.lr_decay_factor < 1 and config.lrd_decay_step > 0:
            optim_tag = '{}_lrd{}-{}'.format(optim_tag, config.lr_decay_step, config.lr_decay_factor)   
        config.optim_tag = optim_tag
        config.with_virtual_pts = True if 'epipolar' in config.loss_type else False

    # Setup network prediction type: ess vector or relative pose
    config.pair_type = 'ess' if 'RelaPose' not in config.network else 'relapose'
    return config
    
class RelaPoseConfig:
    def __init__(self):
        description = 'Relative Pose Estimation'   
        parser = argparse.ArgumentParser(description=description)
        self.parser = parser
        
        # Add different groups for arguments
        prog_group = parser.add_argument_group('General Program Config') 
        data_group = parser.add_argument_group('Data Loading Config', 
                                               'Options regarding image loading and preprocessing')
        model_group = parser.add_argument_group('Model Config', 
                                                'Options regarding network model and optimization')
        visdom_group = parser.add_argument_group('Visdom Config', 
                                                 'Options regarding visdom server for visualization')

        # Program general settings 
        prog_group.add_argument('--test', action='store_false', dest='training', help='set program to a testing phase')
        prog_group.add_argument('--train', action='store_true', dest='training', help='set program to a training phase')     
        prog_group.add_argument('--validate', '-val', metavar='N', type=int,
                                    help='evaluate model every N epochs during training')  
        prog_group.add_argument('--pretrained', metavar='%s',  type=str, default=None,
                                    help='the pre-trained weights to initialize the model(default: %(default)s)')
        prog_group.add_argument('--resume', metavar='%s',  type=str, default=None, 
                                    help='the checkpoint file to reload(default: %(default)s)')
        prog_group.add_argument('--seed', metavar='%d', type=int, default=1, 
                                    help='seed for randomization(default: %(default)s)')
        prog_group.add_argument('--odir', metavar='%s', type=str, required=True, 
                                    help='directory for program outputs')
        prog_group.add_argument('--gpu', metavar='%d',  type=int, default=0, 
                                    help='gpu device to use(cpu used if no available gpu)(default: %(default)s)')
        prog_group.add_argument('--ransac_thres', '-rthres', metavar='%d', type=int,  nargs='+', default=[5],
                                    help='the set of ransac inlier thresolds(angle error)(default: %(default)s)')

        # Data loading and preprocess
        data_group.add_argument('--data_root', metavar='%s', type=str, default='data/',
                                    help='the dataset root directory(default: %(default)s)' )
        data_group.add_argument('--datasets', '-ds', metavar='%s', type=str, nargs='+', 
                                    help="list of training datasets, opts: "\
                                         "['CambridgeLandmarks', '7Scenes'].")       
        data_group.add_argument('--batch_size', '-b', metavar='%d', type=int, default=16, 
                                    help='batch size to load the image data(default: %(default)s)')
        data_group.add_argument('--num_workers', '-n', metavar='%d', type=int, default=0, 
                                    help='number of workers for data loading(default: %(default)s)')        
        data_group.add_argument('--rescale', '-rs', metavar='%d', type=int, default=480,
                                    help='rescale the smaller edge of the image to (default: %(default)s)')
        data_group.add_argument('--crop', metavar='%d', type=int, default=448,
                                    help='cropping size(default: %(default)s).')
        data_group.add_argument('--normalize', '-norm', action='store_true',
                                    help='normalize images with imagenet mean&std')
        data_group.add_argument('--incl_sces', metavar='%s', type=str, nargs='+', 
                                    help="only use specified scenes of the dataset, "\
                                         "by default all scenes of that dataset are used."\
                                         "Not applicable to multi-datasets case!")
        data_group.add_argument('--abs_label_txt', metavar=('%s[train_txt]', '%s[test_txt]'),
                                    nargs=2, default=['dataset_train.txt', 'dataset_test.txt'], 
                                    help='absolute pose label files(default: %(default)s)')
        data_group.add_argument('--pair_txt', '-pair', metavar='%s', type=str, 
                                    default='train_pairs.visloc.txt', 
                                    help='relative pose pair file(default: %(default)s)')
        data_group.add_argument('--val_pair_txt', '-vpair', metavar='%s', type=str, 
                                    default='val_pairs.visloc.txt', 
                                    help='relative pose validation pair file(default: %(default)s)')         
        data_group.add_argument('--with_ess', '-ess', action='store_true', 
                                    help='include essential matrices into data batches')
        
        # Model training
        model_group.add_argument('--network', type=str, default='EssNet', 
                                     choices=['EssNet', 'NCEssNet', 'RelaPoseNet', 'RelaPoseMNet', 'EssNetConcat'], 
                                     help='network architecture(default: %(default)s)')        
        model_group.add_argument('--ess_proj', action='store_true', 
                                     help='project predicted essential matrices(default: %(default)s)')        
        model_group.add_argument('--early_feat', action='store_true', 
                                     help='stop the feature extraction one layer eariler,' \
                                          'meaning network downscales the input by 16 otherwise 32')
        model_group.add_argument('--epochs', '-ep', metavar='%d', type=int, default=200, 
                                     help='number of training epochs(default: %(default)s)')
        model_group.add_argument('--lr_init', '-lr', metavar='%f', type=float, default=1e-4,
                                     help='initial learning rate(default: %(default)s)') 
        model_group.add_argument('--lr_decay', '-lrd', metavar=('%f[factor]', '%d[step]'), nargs=2, default=[1.0, -1],
                                     help='learning rate decay factor and step(default: %(default)s)')
        model_group.add_argument('--weight_decay', '-wd', metavar='%f', type=float, default=1e-6, 
                                     help='weight decay rate(default: %(default)s)')        

        model_group.add_argument('--loss_type', type=str, default='mse', 
                                     choices=['mse', 'homo_mse', 'beta_mse', 'signed_mse', 'epipolar'],
                                     help='loss types(default: %(default)s)')
        model_group.add_argument('--beta',  metavar='%s', type=int, default=1, 
                                     help='beta value in loss type beta_mse for relapose regression(default: %(default)s)')
        model_group.add_argument('--homo', metavar=('Sx', 'Sq'), type=float, nargs=2, default=[0.0, -3.0], 
                                     help='initial weighting variables in loss type homo_mse for '\
                                          'relapose regression(default: %(default)s)')

        # Visdom server setting for visualization
        visdom_group.add_argument('--visenv', '-venv', metavar='%s', type=str, default=None, 
                                      help='the environment for visdom to save all data(default: %(default)s)')
        visdom_group.add_argument('--viswin', '-vwin', metavar='%s', type=str, default=None, 
                                      help='the prefix appended to window title(default: %(default)s)')
        visdom_group.add_argument('--visport', '-vp', metavar='%d', type=int, default=9333, 
                                      help='the port where the visdom server is running(default: %(default)s)')
        visdom_group.add_argument('--vishost', '-vh', metavar='%s', type=str, default='localhost', 
                                      help='the hostname/ip where the visdom server is running(default: %(default)s)')

    def parse(self):
        args = self.parser.parse_args()
        config = setup_config(args)
        return config
        
class RPEvalConfig:
    def __init__(self, data_root, datasets=[], incl_sces=None, 
                 pair_txt=None, rescale=480, crop=448, normalize=True, 
                 network='EssNet', ess_proj=True, early_feat=False,
                 resume=None, pretrained=None, ransac_thres =[5], gpu=0, odir=None):
        
        self.seed = 1
        self.gpu = gpu
        self.batch_size = 16
        self.num_workers = 0
        self.training = False
        self.ransac_thres = ransac_thres
        
        # Load model weights
        self.network = network        
        self.early_feat = early_feat
        self.ess_proj = ess_proj
        self.resume = resume
        self.pretrained = pretrained
        
        # Define image preprocessing
        self.data_root = data_root
        self.datasets = datasets
        self.incl_sces = incl_sces
        self.rescale = rescale
        self.crop = crop
        self.normalize = normalize
        self.abs_label_txt = ['dataset_train.txt', 'dataset_test.txt']
        self.pair_txt = pair_txt
        self.with_ess = False
        
        # Logging
        self.odir = odir
        setup_config(self)

    def __repr__(self):
        return config2str(self)
        
if __name__ == '__main__':
    conf = RelaPoseConfig().parse()

