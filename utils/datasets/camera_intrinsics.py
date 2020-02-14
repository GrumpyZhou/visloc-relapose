import os
import numpy as np

def get_camera_intrinsic_loader(base_dir, dataset, scene, wt, ht):
    if 'Cambridge' in dataset:
        return CambridgeIntrinsics(base_dir, scene, wt=wt, ht=ht)
    elif '7Scenes' in dataset:
        return Scenes7Intrinsics(base_dir, scene, wt=wt, ht=ht)
    
class CambridgeIntrinsics:
    scenes = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']
    def __init__(self, base_dir, scene, wt=1920, ht=1080, w=1920, h=1080):
        assert scene in self.scenes
        self.base_dir = base_dir
        self.scene = scene
        self.wt, self.ht = wt, ht
        self.w, self.h = w, h
        self.ox, self.oy = w / 2, h / 2
        self.sK = np.array([[wt / w, 0, 0],
                            [0, ht / h, 0],
                            [0, 0, 1]])
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
            K = np.array([[f, 0, self.ox], 
                          [0, f, self.oy], 
                          [0, 0, 1]], dtype=np.float32)
            K = self.sK.dot(K)
            matrices[im] = K 
        return matrices   
    
    def get_relative_intrinsic_matrix(self, im1, im2):
        f1 = self.focals[im1]
        f2 = self.focals[im2]        
        f = (f1 + f2) / 2.0    # Compute mean focal of two images
        K = np.array([[f, 0, self.ox], 
                      [0, f, self.oy], 
                      [0, 0, 1]])
        K = self.sK.dot(K)        
        return K

class Scenes7Intrinsics:
    scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    def __init__(self, base_dir, scene, wt=640, ht=480, w=640, h=480):
        assert scene in self.scenes
        self.base_dir = base_dir
        self.scene = scene
        self.wt, self.ht = wt, ht
        self.w, self.h = w, h
        self.ox, self.oy = w / 2, h / 2
        self.sK = np.array([[wt / w, 0, 0],
                            [0, ht / h, 0],
                            [0, 0, 1]])
        self.f = 585
        self.K = self.sK.dot(np.array([[self.f, 0, self.ox], [0, self.f, self.oy], [0, 0, 1]], dtype=np.float32))

    def get_intrinsic_matrices(self, im_list=None):
        if im_list is None:
            return self.K
        else:
            return {im:self.K for im in im_list}
    
    def get_relative_intrinsic_matrix(self, *args):
        return self.K

class PragueIntrinsics:
    scenes = ['castle_paradise_garden', 'castle_yard', 'charles_square_station_escalator',
              'dancing_house', 'st_georges_basilica_building', 'st_vitus_cathedral',
              'street_facades', 'zurich_eth_hg_rows']
    @staticmethod
    def get_imsize(base_dir, scene):        
        intrinsics = os.path.join(base_dir, scene,'intrinsics.txt')
        ff = open(intrinsics, 'r')
        params = ff.readlines()[3].split()
        w, h = [float(p) for p in params[2:4]]
        ff.close()
        return w, h            

    def __init__(self, base_dir, scene, wt=1557, ht=642):
        assert scene in self.scenes
        self.base_dir = base_dir
        self.scene = scene
        intrinsics = os.path.join(self.base_dir, self.scene,'intrinsics.txt')
        ff = open(intrinsics, 'r')
        params = ff.readlines()[3].split()
        self.w, self.h, self.fx, self.fy = [float(p) for p in params[2:6]]
        ff.close()
        
        self.wt, self.ht = wt, ht        
        self.ox, self.oy = self.w / 2, self.h / 2
        self.sK = np.array([[self.wt / self.w, 0, 0],
                            [0, self.ht / self.h, 0],
                            [0, 0, 1]])        
        self.K = self.sK.dot(np.array([[self.fx, 0, self.ox], [0, self.fy, self.oy], [0, 0, 1]], dtype=np.float32))        

    def get_intrinsic_matrices(self, im_list=None):
        if im_list is None:
            return self.K
        else:
            return {im:self.K for im in im_list}
    
    def get_relative_intrinsic_matrix(self, *args):
        return self.K
    
if __name__ == '__main__': 
    loader = CambridgeLandmarks(base_dir='../../data/datasets_original/CambridgeLandmarks/', scene='ShopFacade')
    loader.get_relative_intrinsics('seq3/frame00008.png', 'seq3/frame00003.png')
    
    loader = Scenes7(base_dir='../../data/datasets_original/7Scenes', scene='heads')
    loader.get_relative_intrinsics('seq3/frame00008.png', 'seq3/frame00003.png')