import torch
import random
import numpy as np
from collections import OrderedDict

def load_weights(weights_dir, device):
    map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
    weights_dict = None
    if weights_dir is not None: 
        weights = torch.load(weights_dir, map_location=map_location)
        if isinstance(weights, OrderedDict):
            weights_dict = weights
        elif isinstance(weights, dict) and 'state_dict' in weights:
            weights_dict = weights['state_dict']
    return weights_dict

def lprint(ms, log=None):
    '''Print message on console and in a log file'''
    print(ms)
    if log:
        log.write(ms+'\n')
        log.flush()
        
def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Important also

def config2str(config):
    print_ignore = ['weights_dict', 'optimizer_dict']
    args = vars(config)
    separator = '\n' 
    confstr = ''
    confstr += '------------ Configuration -------------{}'.format(separator)
    for k, v in sorted(args.items()):
        if k in print_ignore:
            if v is not None:
                confstr += '{}:{}{}'.format(k, len(v), separator)
            continue
        confstr += '{}:{}{}'.format(k, str(v), separator)
    confstr += '----------------------------------------{}'.format(separator)
    return confstr

def vec2str(vec, end=''):
    """Convert a 1D vector to string
    Values are seperated by an empty space.
    An ending token is appended at the end of the string. 
    Default ending token is empty.
    Example:
       vec: (1, 2, 3) 
       output: '1 2 3'
    """
    
    assert isinstance(vec, (np.ndarray, list, tuple))
    if isinstance(vec, np.ndarray):
        vec = np.squeeze(vec)
    str_vec = ''
    for i, v in enumerate(vec):
        str_vec += '{:.8f}'.format(v)
        if i != len(vec)-1:
            str_vec += ' '
    str_vec = str_vec + end
    return str_vec


def array2str(arr, end=''):
    """Convert a 2D array to string
    Values are seperated by an empty space.
    Each row is seperated by '\n'
    An ending token is appended at the end of the string. 
    Default ending token is empty.
    Example:
       arr: [[1, 2, 3],
             [4, 5, 6]]
       output: '1 2 3\n4 5 6\n'
    """
    
    assert isinstance(arr, np.ndarray)
    rows = arr.shape[0]
    str_arr = ''
    for i in range(rows):
        str_arr += vec2str(arr[i], end='\n')
    str_arr = str_arr + end
    return str_arr