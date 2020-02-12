import os
import numpy as np

def get_camera_intrinsic_loader(base_dir, dataset, scene):
    if 'Cambridge' in dataset:
        return CambridgeLandmarks(base_dir, scene)
    elif '7Scenes' in dataset:
        return Scenes7(base_dir, scene)
    
class CambridgeLandmarks:
    scenes = ['ShopFacade', 'KingsCollege', 'OldHospital', 'StMarysChurch']
    def __init__(self, base_dir, scene):
        assert scene in self.scenes
        self.base_dir = base_dir
        self.scene = scene
        self.w, self.h = 1920, 1080
        self.ox, self.oy = 960, 540
        self.focals = self.get_focals()

    def get_focals(self):
        focals = {}
        nvm = os.path.join(self.base_dir, self.scene,'reconstruction.nvm')
        with open(nvm, 'r') as f:
            # Skip headding lines
            next(f)   
            next(f)
            cam_num = int(f.readline().split()[0])
            print('Loading focals scene: {} cameras: {}'.format(self.scene, cam_num))
            
            focals = {}
            for i in range(cam_num):
                line = f.readline()
                cur = line.split()
                focals[cur[0].replace('jpg', 'png')] = float(line.split()[1])
        return focals
    
    def get_intrinsic_matrices(self, im_list=None):
        matrices = {}
        if im_list is None:
            im_list = list(self.focals.keys())
        for im in im_list:
            f = self.focals[im]
            K = np.array([[f, 0, self.ox], [0, f, self.oy], [0, 0, 1]], dtype=np.float32) 
            matrices[im] = K
        return matrices
    
    def get_relative_intrinsic_matrix(self, im1, im2):
        f1 = self.focals[im1]
        f2 = self.focals[im2]        
        f = (f1 + f2) / 2.0    # Compute mean focal of two images
        fx, fy = f, f
        K = np.array([[f, 0, self.ox], [0, f, self.oy], [0, 0, 1]]) 
        return K

class Scenes7:
    scenes = ['heads', 'chess', 'fire', 'office', 'pumpkin', 'redkitchen', 'stairs']
    def __init__(self, base_dir, scene):
        assert scene in self.scenes
        self.base_dir = base_dir
        self.scene = scene
        self.w, self.h = 640, 480
        self.ox, self.oy = 320, 240
        self.f = 585
        self.K = np.array([[self.f, 0, self.ox], [0, self.f, self.oy], [0, 0, 1]], dtype=np.float32) 

    def get_intrinsic_matrices(self, im_list):
        matrices = {im :self.K for im in im_list}
        return matrices
    
    def get_relative_intrinsic_matrix(self, *args):
        return self.K

class Prague:
    scenes = ['castle_paradise_garden', 'castle_yard', 'charles_square_station_escalator',
              'dancing_house', 'st_georges_basilica_building', 'st_vitus_cathedral',
              'street_facades', 'zurich_eth_hg_rows']
    
    def __init__(self, base_dir, scene):
        assert scene in self.scenes
        self.base_dir = base_dir
        self.scene = scene
        intrinsics = os.path.join(self.base_dir, self.scene,'intrinsics.txt')
        ff = open(intrinsics, 'r')
        params = ff.readlines()[3].split()
        self.w, self.h, self.fx, self.fy, self.ox, self.oy = [float(p) for p in params[2::]]
        ff.close()
        self.K = np.array([[self.fx, 0, self.ox], [0, self.fy, self.oy], [0, 0, 1]], dtype=np.float32) 

    def get_intrinsic_matrices(self, im_list):
        matrices = {im : self.K for im in im_list}
        return matrices
    
    def get_relative_intrinsic_matrix(self, *args):
        return self.K
    
if __name__ == '__main__': 
    loader = CambridgeLandmarks(base_dir='../../data/datasets_original/CambridgeLandmarks/', scene='ShopFacade')
    loader.get_relative_intrinsics('seq3/frame00008.png', 'seq3/frame00003.png')
    
    loader = Scenes7(base_dir='../../data/datasets_original/7Scenes', scene='heads')
    loader.get_relative_intrinsics('seq3/frame00008.png', 'seq3/frame00003.png')