import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.utils.data as data
from utils.datasets.data_parsing import *
from utils.datasets.camera_intrinsics import get_camera_intrinsic_loader

__all__ = ['glob_scenes', 'get_datasets', 'VisualLandmarkDataset']

def glob_scenes(data_root, pair_txt):
    import glob
    scenes = []
    for sdir in glob.iglob('{}/*/{}'.format(data_root, pair_txt)):
        sdir = sdir.split('/')[-2]  
        scenes.append(sdir)
    return sorted(scenes)

def get_scene_dataset(dataset, scene, pair_txt, data_root='data', ops=None, 
                     train_lbl_txt=None, test_lbl_txt=None, with_ess=True,
                     with_virtual_pts=False):
    print('>>>Load Dataest: {} Scene {} Data root: {}'.format(dataset, scene, data_root))
    
    # Init dataset objects for the scene
    data_src = VisualLandmarkDataset(data_root, dataset, scene, pair_txt, 
                                     with_ess, train_lbl_txt, test_lbl_txt, ops,
                                     with_virtual_pts=with_virtual_pts)
    return data_src


def get_datasets(datasets, pair_txt,  data_root='data', incl_sces=None, ops=None, 
                 train_lbl_txt=None, test_lbl_txt=None, with_ess=True,
                 with_virtual_pts=False):
    dataset_opts = ['CambridgeLandmarks', '7Scenes', 'TUMLSI', 'MegaDepth', 'ScanNet', 'PragueZurich', 'DeepLoc', 'IDL']
    print('>>>Load dataest: {} Data root: {}'.format(datasets, data_root))
    data_srcs = []
    for dataset in datasets: 
        sces = []
        if dataset not in dataset_opts:
            print('Dataset {} does not exist, please check the dataset name!'.format(dataset))
            continue
        if not incl_sces or len(datasets) > 1:
            sces = glob_scenes(os.path.join(data_root, dataset), pair_txt)    # Locate all scenes of the current dataset
        else:
            sces = incl_sces
        print('Dataset: {} Scene: {}\n'.format(dataset, sces))

        # Init dataset objects for each scene
        for scene in sces:
            data_src = VisualLandmarkDataset(data_root, dataset, scene, pair_txt, 
                                             with_ess, train_lbl_txt, test_lbl_txt, ops,
                                             with_virtual_pts=with_virtual_pts)
            data_srcs.append(data_src)
    return data_srcs

class RelaPoseDataset(data.Dataset):
    def __init__(self, data_dir, pose_pairs, transforms=None, with_virtual_pts=False, intrinsics=None):
        self.data_dir = data_dir
        self.transforms = transforms 
        self.im_pairs = pose_pairs.im_pairs 
        self.ess_vecs = pose_pairs.ess_vecs
        self.relv_poses = pose_pairs.relv_poses        
        self.abs_poses = pose_pairs.abs_poses
        self.num = pose_pairs.num
        self.with_virtual_pts = with_virtual_pts
        self.intrinsics = intrinsics
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict:  'im_pairs' is a tuple of two image tensors,
                   'relv_poses' is the relative pose between the two images,
                   'ess_vec' is the essential vector of the two images or None if not provided,                   
                   'train_abs_c' is absolute position of the 1st image poses,
                   'test_abs_q' is absolute rotation of the 1st image poses,
                   'train_abs_c' is absolute position of the 2st image poses,
                   'test_abs_q' is absolute rotation of the 2st image poses,
        """
        data_dict = {}
        im1_ref, im2_ref = self.im_pairs[index]
        data_dict['im_pair_refs'] = (im1_ref, im2_ref)
        im1 = Image.open(os.path.join(self.data_dir, im1_ref))
        im2 = Image.open(os.path.join(self.data_dir, im2_ref))
        if self.transforms:
            im1, im2 = self.transforms(im1, im2)
        abs_lbl = self.abs_poses[index] if self.abs_poses else None
        data_dict['im_pairs'] = (im1, im2)
        if self.ess_vecs:
            data_dict['ess_vec'] = self.ess_vecs[index]
        data_dict['relv_t'] = self.relv_poses[index][0]
        data_dict['relv_q'] = self.relv_poses[index][1]
        if self.abs_poses:
            data_dict['train_abs_c'], data_dict['train_abs_q'] =  self.abs_poses[index][0]          
            data_dict['test_abs_c'], data_dict['test_abs_q'] = self.abs_poses[index][1]
         
        if self.with_virtual_pts:
            K1, K1_inv = self.intrinsics[im1_ref]
            K2, K2_inv = self.intrinsics[im2_ref]
            data_dict['intrinsics'] = (K1, K2, K1_inv, K2_inv)
            matches, F = get_virtual_matches(data_dict['ess_vec'], data_dict['intrinsics'])
            data_dict['virtual_pts'] = matches
            data_dict['F_mats'] = F
        return data_dict

    def __len__(self):
        return self.num

class VisualLandmarkDataset(RelaPoseDataset):    
    def __init__(self, data_root, dataset, scene, pair_txt, with_ess=True, train_lbl_txt=None, test_lbl_txt=None, transforms=None, with_virtual_pts=False):
        self.dataset=dataset
        self.scene = scene
        self.pair_txt = os.path.join(data_root, dataset, scene, pair_txt)
        self.train_lbl_txt = None if not train_lbl_txt else os.path.join(data_root, dataset, scene, train_lbl_txt)
        self.test_lbl_txt = None if not train_lbl_txt else os.path.join(data_root, dataset, scene, test_lbl_txt)
        pose_pairs = make_pose_pairs(self.pair_txt, with_ess, self.train_lbl_txt, self.test_lbl_txt)
        
        # Loading virtual
        intrinsics = None
        if with_virtual_pts:
            intrinsics_loader = get_camera_intrinsic_loader(os.path.join(data_root, dataset),
                                                            dataset, scene)
            intrinsics = intrinsics_loader.get_intrinsic_matrices(im_list=list(pose_pairs.pose_dict.keys()))
                                                                  
        super().__init__(os.path.join(data_root, dataset, scene), pose_pairs, 
                         transforms=transforms, with_virtual_pts=with_virtual_pts, intrinsics=intrinsics)

    def __repr__(self):
        fmt_str = 'Dataset VisualLandmarkDataset - {} - {}\n'.format(self.dataset, self.scene)
        fmt_str += 'Number of data pairs: {}\n'.format(self.__len__())
        fmt_str += 'Root location: {}\n'.format(self.data_dir)
        fmt_str += 'Pair txt: {}\n'.format(self.pair_txt)
        fmt_str += 'Train label txt: {}\n'.format(self.train_lbl_txt)
        fmt_str += 'Test label txt: {}\n'.format(self.test_lbl_txt)
        fmt_str += 'Transforms: {}\n'.format(self.transforms.__repr__().replace('\n', '\n    '))
        return fmt_str

class PosePairs:
    def __init__(self, im_pairs, relv_poses, ess_vecs=None, abs_poses=None, pose_dict=None):
        self.im_pairs = im_pairs
        self.relv_poses = relv_poses
        self.ess_vecs = ess_vecs
        self.abs_poses = abs_poses
        self.pose_dict = pose_dict
        self.num = len(self.im_pairs)

def make_pose_pairs(pair_txt, with_ess, train_lbl_txt=None, test_lbl_txt=None):
    im_pairs, relv_poses, ess_vecs = parse_relv_pose_txt(pair_txt, with_ess)
    abs_poses = None
    pose_dict = None
    if train_lbl_txt and test_lbl_txt:
        abs_poses = []
        pose_dict = parse_abs_pose_txt(train_lbl_txt)
        pose_dict.update(parse_abs_pose_txt(test_lbl_txt))
        for pair in im_pairs:
            abs_poses.append((pose_dict[pair[0]], pose_dict[pair[1]]))
    return PosePairs(im_pairs, relv_poses, ess_vecs, abs_poses, pose_dict)

def get_virtual_matches(ess_vec, intrinsic, vgrid_step=0.02):
    """Calculate inlier matches of an essential matrice
    First calcuate Fundamental matrix  from essential matrix and the camera intrinsics
    Fit matches using cv.correctMatches()
    
    Args:
        ess_vec: the essential matrice
        intrinsices: a tuple of 4 (K1, K2, K1_inv, K2_inv), the camera calibriation matrices and their inverse 
        vgrid_step: the resolution to calculate the virtual points grids that are used into for fitting
    
    Return:
        matches: the array consists of (1 / vgrid_step) ** 2 ) point correspondences
        F: the Fundamental matrix
    """
    
    xx, yy = np.meshgrid(np.arange(0, 1 , vgrid_step), np.arange(0, 1, vgrid_step))
    num_vpts = int((1 / vgrid_step) ** 2 ) 
    E = ess_vec.reshape((3,3))
    K1, K2, K1_inv, K2_inv = intrinsic
    F = K2_inv.T @ E @ K1_inv
    F = F / np.linalg.norm(F)    # Normalize F

    w1, h1 = 2 * K1[0, 2], 2 * K1[1, 2]
    w2, h2 = 2 * K2[0, 2], 2 * K2[1, 2]    

    pts1_vgrid = np.float32(np.vstack((w1 * xx.flatten(), h1 * yy.flatten())).T)
    pts2_vgrid = np.float32(np.vstack((w2 * xx.flatten(), h2 * yy.flatten())).T)
    pts1_virt, pts2_virt = cv2.correctMatches(F, np.expand_dims(pts1_vgrid, axis=0), np.expand_dims(pts2_vgrid, axis=0))

    nan1 = np.logical_and(
            np.logical_not(np.isnan(pts1_virt[:,:,0])),
            np.logical_not(np.isnan(pts1_virt[:,:,1])))
    nan2 = np.logical_and(
            np.logical_not(np.isnan(pts2_virt[:,:,0])),
            np.logical_not(np.isnan(pts2_virt[:,:,1])))

    _, pids = np.where(np.logical_and(nan1, nan2))
    num_good_pts = len(pids)
    while num_good_pts < num_vpts:
        pids = np.hstack((pids, pids[:(num_vpts - num_good_pts)]))
        num_good_pts = len(pids)

    pts1_virt = pts1_virt[:, pids]
    pts2_virt = pts2_virt[:, pids]

    # Homogenous
    ones = np.ones((pts1_virt.shape[1],1))
    pts1_virt = np.hstack((pts1_virt[0], ones))
    pts2_virt = np.hstack((pts2_virt[0], ones))
    matches = (pts1_virt.astype(np.float32), pts2_virt.astype(np.float32))
    return matches, F