import numpy as np
import torch
import math

########################
## Pytorch Interfaces ##
########################

def vec_ang_err(v1, v2, eps=1e-10):
    """Calculate angular error between two vectors
    Input:
        v1: a tensor of 3D vectors, size (N, 3)
        v2: another tensor of 3D vectors, size (N, 3)
        eps: small epsilon to prevent nan value when the cosine value is exatly 1.0
    Return:
        theta: a tensor of angles error in degree, size(N,)
    """
    cosd = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
    #d = torch.clamp(cosd, min=-1.0+eps, max=1.0-eps)  # Problematic
    theta = torch.acos(cosd) * 180 / math.pi
    return theta

def quat_ang_err(q1, q2, eps=1e-10):
    """Calculate angular error between two quaternions
    Input:
        q1: a tensor of 4D quaternions, size (N, 4)
        q2: another tensor of 4D quaternions, size (N, 4)
        eps: small epsilon to prevent nan value when the cosine value is exatly 1.0
    Return:
        theta: a tensor of angles error in degree, size(N,)
    """
    cosd = torch.nn.functional.cosine_similarity(q1, q2, dim=-1)
    #d = torch.clamp(cosd.abs(), min=-1.0+eps, max=1.0-eps)
    d = cosd.abs()
    theta = 2 * torch.acos(d) * 180 / math.pi
    return theta

def vec_cosine_err(v1, v2):
    """Calculate cosine similarity between two vectors
    Input:
        v1: a tensor of 3D vectors, size (N, 3)
        v2: another tensor of 3D vectors, size (N, 3)
    Return:
        dist: a tensor of cosine similarity values, size(N,)
    """
    dist = 1.0 - 1.0 * torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
    return dist

def quat_cosine_err(q1, q2):
    """Calculate cosine similarity between two quaternions
    Input:
        q1: a tensor of 4D quaternions, size (N, 4)
        q2: another tensor of 4D quaternions, size (N, 4)
    Return:
        dist: a tensor of cosine similarity values, size(N,)
    """
    dist = 1.0 - 1.0 * torch.nn.functional.cosine_similarity(q1, q2, dim=-1) # TODO: double check whether we need this abs
    return dist

######################
## Numpy Interfaces ##
######################

def cal_vec_angle_error(label, pred, eps=1e-10):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
        
    # TODO: now not adding eps just to check whether zero translation happens very often
    # Later ignore these cases in testing?
    #v1 = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), eps)
    #v2 = label / np.maximum(np.linalg.norm(label, axis=1, keepdims=True), eps)
    v1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    v2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.around(np.sum(np.multiply(v1,v2), axis=1, keepdims=True), decimals=4)  # around to 0.0001 can assure d <= 1    
    error = np.degrees(np.arccos(d))
    error[np.where(np.isnan(error))] = 0.0 # TODO: quick fix for zero position which leads to zero division, currently set the case to error = 0
    return error

def cal_quat_angle_error(label, pred):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.abs(np.sum(np.multiply(q1,q2), axis=1, keepdims=True)) # Here we have abs()!
    d = np.clip(d, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d))
    return error
