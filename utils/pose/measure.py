import numpy as np
import torch
import math

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
