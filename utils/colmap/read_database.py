import sys
import sqlite3
import numpy as np

from utils.colmap.bases import *

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

def extract_pair_pts(pair_id, key_points, matches):
    """Get point correspondences of a pair
    
    Args:
        pair_id: tuple (im1_id, im2_id)
        key_points: dict {image_id: keypoints}
        matches: dict {image_pair_id: (key_pt_ids1, key_pt_ids2)}
    Return:
        pts: contains pixel locations of all correspondences, stored as Nx4 numpy array.
               Four columns are x1, y1, x2, y2.
               If there's no correspondences detected for this pair, None is returned, 
        invalid: set to 0 if there exists no correspondences for this pair, otherwise 1.
    """
    invalid = 0
    (im1, im2) = pair_id
    kpts1 = key_points[im1]
    kpts2 = key_points[im2]
    if (im1, im2) not in matches:
        invalid = 1
        return None, invalid
    key_ids = matches[(im1, im2)]
    num_pts = key_ids.shape[0]
    pts = np.zeros((num_pts, 4))
    for j in range(num_pts):
        k1, k2 = key_ids[j,:]
        x1, y1 = kpts1[k1][0:2]
        x2, y2 = kpts2[k2][0:2]
        pts[j, :] = [x1, y1, x2, y2]
    return pts, invalid

def extract_all_pair_pts(pair_ids, key_points, matches):
    '''Get pixel location of correspondence'''
    corrd_dict = {}
    invalid = [] 
    for i, (i1, i2) in enumerate(pair_ids):
        feat1 = key_points[i1]
        feat2 = key_points[i2]
        if (i1, i2) in matches:
            fmatch = matches[(i1, i2)]
        else:
            invalid.append(i)
            corrds = None
            continue
        Nf = fmatch.shape[0]
        corrds = np.zeros((Nf, 4))
        for j in range(Nf):
            # Get pixel location
            k1, k2 = fmatch[j,:]
            x1, y1 = feat1[k1][0:2]
            x2, y2 = feat2[k2][0:2]
            corrds[j, :] = [x1, y1, x2, y2]
        corrd_dict[(i1, i2)] = corrds
    return corrd_dict, invalid

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return int(image_id1), int(image_id2)

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)        
    

class COLMAPDataLoader:
    def __init__(self, database_path):
        super(COLMAPDataLoader, self).__init__()
        self.db = sqlite3.connect(database_path)
        self.cameras = None
        self.images_name_based = None
        self.images_id_based = None
        self.keypoints = None
        self.matches = None
        self.descriptors = None
        
    def load_images(self, name_based=False):
        if not self.images_id_based or not self.images_name_based:
            images_name_based = {}
            images_id_based = {}
            for image_id, name, camera_id in self.db.execute("SELECT image_id, name, camera_id FROM images"):
                images_name_based[name] = (image_id, camera_id)
                images_id_based[image_id] = (name, camera_id)  
            self.images_name_based = images_name_based
            self.images_id_based = images_id_based
            print('Load images to dataloader')
        return self.images_name_based if name_based else self.images_id_based

    def load_cameras(self,):
        if not self.cameras:
            cameras = {}
            for row in self.db.execute("SELECT * FROM cameras"):
                camera_id, model_id, width, height, params, prior = row
                model_name = CAMERA_MODEL_IDS[model_id].model_name
                #num_params = CAMERA_MODEL_IDS[model_id].num_params
                params = blob_to_array(params, np.float64)
                CameraParam = CAMERA_PARAMS[model_name]
                cameras[camera_id] = Camera(id=camera_id,
                                            model=model_name,
                                            width=width,
                                            height=height,
                                            params=CameraParam._make(params))
            self.cameras = cameras
            print('Load cameras to dataloader')
        return self.cameras
    
    def load_descriptors(self):
        if not self.descriptors:
            descriptors = dict(
                (image_id, blob_to_array(data, np.uint8, (-1, 128)))
                for image_id, data in self.db.execute(
                    "SELECT image_id, data FROM descriptors"))
            self.descriptors = descriptors
            print('Load descriptors to dataloader')
        return self.descriptors

    def load_keypoints(self, key_len=6):
        """
        Note that COLMAP supports:
         - 2D keypoints: (x, y)
         - 4D keypoints: (x, y, scale, orientation)
         - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

        Return:
            keypoints: dict {image_id: keypoints}
        """
        
        if not self.keypoints:
            keypoints = dict(
                (image_id, blob_to_array(data, np.float32, (-1, key_len)))
                for image_id, data in self.db.execute(
                    "SELECT image_id, data FROM keypoints"))
            self.keypoints = keypoints
            print('Load keypoints to dataloader')
        return self.keypoints
        
    def load_matches(self):
        """Load all matches.
        
        Notice this will take lots of time if there are lots of images
        Return:
            matches: dict {image_pair_id: matches}
        """
        
        if not self.matches:
            matches = {}
            for pair_id, data in self.db.execute("SELECT pair_id, data FROM matches"):
                if data is not None:
                    im_pair_id = pair_id_to_image_ids(pair_id)
                    matches[im_pair_id] = blob_to_array(data, np.uint32, (-1, 2))
            self.matches = matches
            print('Load matches to dataloader')
        return self.matches
    
    def load_pair_matches(self, im_pair_ids):
        '''Load specified matches
        
        Arg:
            im_pair_ids: list of tuple (im1_id, im2_id)
        Return:
            matches: dict {image_pair_id: matches}
        '''
        if not self.matches:
            matches = {}
            for im_pair_id in im_pair_ids:
                im1, im2= im_pair_id
                pair_id = image_ids_to_pair_id(im1, im2)
                data = self.db.execute("SELECT data FROM matches where pair_id={}".format(pair_id)).fetchall()[0][0]
                if data is not None:
                    match_val = blob_to_array(data, np.uint32, (-1, 2))
                    if im1 > im2:
                        match_val = match_val[:,::-1] # swap the indices  
                    matches[(im1, im2)] = match_val
            self.matches = matches
            print('Load matches to dataloader')
        return self.matches

    def get_intrinsics(self, im_name):
        self.load_images(name_based=True)   
        self.load_cameras()
        cid = self.images_name_based[train_im][1]
        camera = self.cameras[cid]
        param = camera.params
        ox, oy = param.ox, param.oy
        if 'f' in param:
            fx, fy = param.f, param.f
        else:
            fx, fy = param.fx, param.fy
        return (fx, fy, ox, oy)
       
            
        
    def load_two_view_geometry(self):
        raise NotImplementedError
