"""
The script contains interfaces for camera pose parametrization as angles
Formulas of related calculation can be found in:
- Representing pose with 5 angles:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
"""
import numpy as np
import torch
from transforms3d.euler import euler2quat, quat2euler

########################
## Pytorch Interfaces ##
########################

def cartes_to_spher(vec):
    """
    Convert a tensor from cartesian coordinates 
    to spherical coordinates(ISO convention)
    Input:
        vec: size (N, 3)
    Return:
        radius: r, size (N,)
        polar angle: theta, size (N,)
        azimuthal angle: phi, size (N,)
    """
    if len(vec.size()) == 1:
        vec = vec.unsqueeze(0)  # Expand the batchsize dimension
    r = torch.norm(vec, dim=1)
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    return theta, phi

def spher_to_cartes(theta, phi):
    """Convert spherical coordinates(ISO convention) to cartesian coordinates
    Input:
        polar angle: theta, size (N,)
        azimuthal angle: phi, size (N,)
        and we use assume all vectors have unit length, i.e., r = 1
    Return:
        vec: a tensor of size (N, 3)
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    vec = torch.stack([x, y, z], dim=-1)
    return vec

def quat_to_euler(q):
    if len(q.size()) == 1:
        q = q.unsqueeze(0)  # Expand the batchsize dimension
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # X-Y-Z order is equivalent to roll-pitch-yaw order
    roll = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    pitch = torch.asin(2 * (q0 * q2 - q3 * q1))
    yaw = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    return roll, pitch, yaw

def euler_to_quat(roll, pitch, yaw):
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    q0 = cy * cr * cp + sy * sr * sp
    q1 = cy * sr * cp - sy * cr * sp
    q2 = cy * cr * sp + sy * sr * cp
    q3 = sy * cr * cp - cy * sr * sp
    q = torch.stack([q0, q1, q2, q3], dim=1)
    return q

def angles_to_pose(angles):
    """
    Input:
        angles: a tensor of size (N, 5), each row contains 5 angles
    Return:
        t: translation vectors converted from spherical coordinates to cartesian coordinates, size (N, 3)
        q: rotation quaternions converted from euler angles to quaternions, size (N, 4)
    """
    a0, a1 = angles[:, 0], angles[:, 1]
    a2, a3, a4 = angles[:, 2], angles[:, 3], angles[:, 4]
    t = spher_to_cartes(theta=a0, phi=a1)
    q = euler_to_quat(roll=a2, pitch=a3, yaw=a4)
    return t, q

def quat_log(q):
    """Apply the logarithm map to quaternions
    Input:
      q: a tensor of quaternions, size (N, 4)
    Return: 
      logq: a tensor of log quaternions, size (N, 3)
    """
    if len(q.size()) == 1:
        q = q.unsqueeze(0) # Expand the batch dimension
    norm = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    logq = q[:, 1:] * torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0)) / norm
    return logq

def quat_exp(logq):
    """Apply exponential map to log quaternions
    Input:
        logq: a tensor of log quaternions, size (N, 3)
    Return:
        q: a tensor of original quaternions, size(N, 4)
    """
    if len(logq.size()) == 1:
        logq = logq.unsqueeze(0) # Expand the batch dimension
    norm = torch.norm(logq, p=2, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    q = logq * torch.sin(norm) / norm
    q = torch.cat((torch.cos(norm), q), dim=1)
    return q

######################
## Numpy Interfaces ##
######################

def vec2angle(vec):
    '''Convert from cartesian coordinates to spherical coordinates(ISO convention)
    Input:
        vec: 3D vector (x, y, z)
    Return:
        (r, theta, phi) where r is radius, thetha is polar angle 
        and phi is azimuthal angle
    '''
    x, y, z = np.squeeze(vec)
    r =  np.linalg.norm(vec)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)    
    return r, theta, phi

def angle2vec(r, theta, phi):
    '''Convert spherical coordinates(ISO convention) to cartesian coordinates
    Input:
        (r, theta, phi) where r is radius, thetha is polar angle 
        and phi is azimuthal angle
    Return:
        vec: 3D vector (x, y, z)
    '''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    vec = np.array([x, y, z], dtype=np.float32)
    return vec

def pose2angs(t, q):
    """Convert camera pose to five angles
    Input:
        t: camera translation and |t| = 1, i.e., the scale is lost
        q: camera rotation in quaternion 
    Return:
        theta, phi, ai, aj, ak
        theta, phi are the two angles of translation in spherical coordinates
        ai, aj, ak are the three euler angles of quterninon
    """
    tnorm = np.linalg.norm(t)
    t = t / tnorm if tnorm != 0 else t
    _, theta, phi = vec2angle(t)
    ai, aj, ak = quat2euler(q)
    angs = np.array([theta, phi, ai, aj, ak], dtype=np.float32)
    return angs

def angs2pose(theta, phi, ai, aj, ak):
    """Convert  five angles to camera pose
    Input:
        theta, phi, ai, aj, ak
        theta, phi are the two angles of translation in spherical coordinates
        ai, aj, ak are the three euler angles of quterninon
    Return:        
        t: camera translation and |t| = 1, i.e., the scale is lost        
        q: camera rotation in quaternion 
    """
    t = angle2vec(1.0, theta, phi)
    q = euler2quat(ai, aj, ak)
    return t, q
