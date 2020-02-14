import os
import numpy as np
import torch.utils.data as data
from utils.common.setup_helper import make_deterministic
from utils.datasets.relapose import get_datasets, get_scene_dataset
from utils.datasets.preprocess import get_pair_transform_ops

def cal_rescale_size(image_size, w, h, k_size=2, scale_factor=1/16):
    # Calculate target image size
    wt = int(np.floor(w/(max(w, h)/image_size)*scale_factor/k_size)/scale_factor*k_size)
    ht = int(np.floor(h/(max(w, h)/image_size)*scale_factor/k_size)/scale_factor*k_size)
    N = wt * ht * scale_factor * scale_factor / (k_size ** 2)
    print('Target size {} Rescale size: (w={},h={}) , matches resolution: {}'.format(image_size, wt, ht, N))
    return wt, ht

def get_test_loaders(image_size, dataset, scenes, pair_txt, data_root, with_virtual_pts=False, shuffle=False):
    print('Load testing data for immatch localization')
    print('Image size {} Dataset {} Scenes {} Pairs {}'.format(image_size, dataset, scenes, pair_txt))
    if 'Cambridge' in dataset:
        return get_cambridge_loaders(data_root=data_root, pair_txt=pair_txt, 
                                     image_size=image_size, incl_sces=scenes, 
                                     with_virtual_pts=with_virtual_pts, shuffle=shuffle)
    elif '7Scenes' in dataset:
        return get_7scenes_loaders(data_root=data_root, pair_txt=pair_txt, 
                                   image_size=image_size, incl_sces=scenes, 
                                   with_virtual_pts=with_virtual_pts)

def get_cambridge_loaders(data_root, pair_txt, image_size=1920, incl_sces=None, with_virtual_pts=False, shuffle=False):
    if incl_sces is None:
        incl_sces = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']
    if image_size is None:
        image_size=1920
    datasets=['CambridgeLandmarks']
    wt, ht = cal_rescale_size(image_size, w=1920, h=1080)
    ops = get_pair_transform_ops(resize=(ht, wt), crop=None, scale=True, normalize=True)
    test_sets = get_datasets(datasets=datasets,
                             pair_txt=pair_txt,
                             data_root=data_root,
                             incl_sces=incl_sces,
                             ops=ops,
                             train_lbl_txt='dataset_train.txt', 
                             test_lbl_txt='dataset_test.txt',
                             with_ess=True,
                             with_virtual_pts=with_virtual_pts)
    data_loaders = {}
    for test_set in test_sets:
        data_loaders[test_set.scene] = data.DataLoader(test_set, batch_size=1, 
                                                       shuffle=shuffle, num_workers=0,
                                                       worker_init_fn=make_deterministic(seed=1))
    return data_loaders

def get_7scenes_loaders(data_root, pair_txt, image_size=640, incl_sces=None, with_virtual_pts=False):
    if incl_sces is None:
        incl_sces = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    if image_size is None:
        image_size=640
    datasets=['7Scenes']
    wt, ht = cal_rescale_size(image_size, w=640, h=480)
    ops = get_pair_transform_ops(resize=(ht, wt), crop=None, scale=True, normalize=True)
    test_sets = get_datasets(datasets=datasets,
                             pair_txt=pair_txt,
                             data_root=data_root,
                             incl_sces=incl_sces,
                             ops=ops,
                             train_lbl_txt='dataset_train.txt', 
                             test_lbl_txt='dataset_test.txt',
                             with_ess=True,
                             with_virtual_pts=with_virtual_pts) 
    data_loaders = {}
    for test_set in test_sets:
        data_loaders[test_set.scene] = data.DataLoader(test_set, batch_size=1, 
                                                       shuffle=False, num_workers=0,
                                                       worker_init_fn=make_deterministic(seed=1))
    return data_loaders