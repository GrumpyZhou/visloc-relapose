import numpy as np
import torch
import math
from transforms3d.quaternions import quat2mat, mat2quat

get_statis = lambda arr: 'Size={} Min={:.2f} Max={:.2f} Mean={:.2f} Median={:.2f}'.format(
                                arr.shape, np.min(arr), np.max(arr), np.mean(arr), np.median(arr))

''' Epipolar geometry functionals'''
skew = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

#  Essential matrix to fundamental matrix 
ess2fund = lambda K1, K2, E: np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
ess2fund_inv = lambda K1_inv, K2_inv, E: K2_inv.T @ E @ K1_inv

# Fundamental matrix to essential matrix
fund2ess = lambda F, K2, K1: K2.T @ F @ K1

# Camera relative pose to fundamental matrix
pose2fund = lambda K1, k2, R, t: np.linalg.inv(K2).T @ R @ K1.T @ skew((K1 @ R.T).dot(t))
pose2fund_inv = lambda K1, K2_inv, R, t: K2_inv.T @ R @ K1.T @ skew((K1 @ R.T).dot(t))

# Normalize fundamental matrix
normF = lambda F: F / F[-1,-1] # Normalize F by the last value
normalize = lambda A:  A / np.linalg.norm(A)

def expand_homo_ones(arr2d, axis=1):
    if axis == 0:
        ones = np.ones((1, arr2d.shape[1]))
    else:
        ones = np.ones((arr2d.shape[0], 1))      
    return np.concatenate([arr2d, ones], axis=axis)

def symmetric_epipolar_distance(pts1, pts2, F):
    """Calculate symmetric epipolar distance between 2 sets of points
    Args:
        - pts1, pts2: points correspondences in the two images, 
          each has shape of (num_points, 2)
        - F: fundamental matrix that fulfills x2^T*F*x1=0, 
          where x1 and x2 are the correspondence points in the 1st and 2nd image 
    Return:
        A vector of (num_points,), containing root-squared epipolar distances
          
    """
    
    # Homogenous coordinates
    pts1 = expand_homo_ones(pts1, axis=1)
    pts2 = expand_homo_ones(pts2, axis=1)

    # l2=F*x1, l1=F^T*x2
    l2 = np.dot(F, pts1.T) # 3,N
    l1 = np.dot(F.T, pts2.T)
    dd = np.sum(l2.T * pts2, 1)  # Distance from pts2 to l2
    d = np.abs(dd) * (1.0 / np.sqrt(l1[0, :] ** 2 + l1[1, :] ** 2) + 1.0 / np.sqrt(l2[0, :] ** 2 + l2[1, :] ** 2))
    return d

'''Error metric computation'''
def cal_vec_angle_error(label, pred, eps=1e-10):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)

    v1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    v2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.around(np.sum(np.multiply(v1,v2), axis=1, keepdims=True), decimals=4)  # around to 0.0001 can assure d <= 1
    d = np.clip(d, a_min=-1, a_max=1)
    error = np.degrees(np.arccos(d))
    
    # If vector are all zero leads to zero division
    # currently set such cases to error = 0.
    error[np.where(np.isnan(error))] = 0.0 
    return error

def cal_quat_angle_error(label, pred):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.abs(np.sum(np.multiply(q1,q2), axis=1, keepdims=True))
    d = np.clip(d, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d))
    return error
