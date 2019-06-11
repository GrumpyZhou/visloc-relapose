import os
import numpy as np

def parse_abs_pose_txt(fpath):
    """Absolute pose label format: 
        3 header lines
        list of samples with format: 
            image x y z w p q r
    """
    
    pose_dict = {}
    f = open(fpath)
    for line in f.readlines()[3::]:    # Skip 3 header lines
        cur = line.split(' ')
        c = np.array([float(v) for v in cur[1:4]], dtype=np.float32)
        q = np.array([float(v) for v in cur[4:8]], dtype=np.float32)
        im = cur[0]
        pose_dict[im] = (c, q)
    f.close()
    return pose_dict

def parse_relv_pose_txt(fpath, with_ess=True):
    '''Relative pose pair format:image1 image2 sim w p q r x y z ess_vec'''
    im_pairs = []
    ess_vecs = [] if with_ess else None
    relv_poses = []
    f = open(fpath)
    for line in f:
        cur = line.split()
        im_pairs.append((cur[0], cur[1]))
        q = np.array([float(i) for i in cur[3:7]], dtype=np.float32)
        t = np.array([float(i) for i in cur[7:10]], dtype=np.float32)
        relv_poses.append((t, q))

        if with_ess:  
            ess_vecs.append(np.array([float(i) for i in cur[10:19]], dtype=np.float32))
    f.close()
    return im_pairs, relv_poses, ess_vecs

def parse_matching_pairs(pair_txt):
    """Get list of image pairs for matching
    Arg:
        pair_txt: file contains image pairs and essential 
        matrix with line format
            image1 image2 sim w p q r x y z ess_vec
    Return:
        list of 3d-tuple contains (q=[wpqr], t=[xyz], essential matrix)
    """
    im_pairs = {}
    f = open(pair_txt)
    for line in f:
        cur = line.split()
        im1, im2 = cur[0], cur[1]
        q = np.array([float(i) for i in cur[3:7]], dtype=np.float32)
        t = np.array([float(i) for i in cur[7:10]], dtype=np.float32)
        ess_mat = np.array([float(i) for i in cur[10:19]], dtype=np.float32).reshape(3,3)
        im_pairs[(im1, im2)] = (q, t, ess_mat)
    f.close()
    return im_pairs