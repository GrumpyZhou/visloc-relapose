import argparse
import os
import numpy as np
import cv2
import time

from utils.common.setup_helper import lprint
from utils.colmap.read_database import COLMAPDataLoader, extract_pair_pts
from utils.datasets.camera_intrinsics import get_camera_intrinsic_loader
from utils.datasets.data_parsing import parse_abs_pose_txt, parse_matching_pairs
from utils.pose.localize import *

def predict_essential_matrix(data_root, dataset, scenes,
                             pair_txt, train_lbl_txt, test_lbl_txt, 
                             colmap_db_root, db_name='database.db',
                             rthres=0.5, log=None):
    
    result_dict = {}
    print('Sift scenes: ',scenes)
    for scene in scenes:
        start_time = time.time()
        base_dir = os.path.join(data_root, dataset)
        db_base_dir = os.path.join(colmap_db_root, dataset)
        scene_dir = os.path.join(base_dir, scene)
        intrinsic_loader = get_camera_intrinsic_loader(base_dir, dataset, scene)
        
        # Load image pairs, pose and essential matrix labels
        abs_pose = parse_abs_pose_txt(os.path.join(scene_dir, train_lbl_txt))
        abs_pose.update(parse_abs_pose_txt(os.path.join(scene_dir, test_lbl_txt)))
        im_pairs = parse_matching_pairs(os.path.join(scene_dir, pair_txt)) # {(im1, im2) : (q, t, ess_mat)}
        pair_names = list(im_pairs.keys())      

        # Loading data from colmap database
        database_path = os.path.join(db_base_dir, scene, db_name)
        db_loader = COLMAPDataLoader(database_path)
        key_points = db_loader.load_keypoints(key_len=6)
        images = db_loader.load_images(name_based=True)   
        pair_ids = [(images[im1][0], images[im2][0]) for im1, im2 in pair_names]
        matches = db_loader.load_pair_matches(pair_ids)
        
        total_time = time.time() - start_time
        print('Scene {} Data loading finished, time:{:.4f}'.format(scene, total_time))
        
        # Calculate essential matrixs per query image
        total_num = len(im_pairs)
        pair_data = {}
        start_time = time.time() 
        no_pts_pairs = []
        for i, im_pair in enumerate(pair_names):
            train_im, test_im = im_pair
            # Dict to save results in a structured way for later RANSAC
            if test_im not in pair_data:
                pair_data[test_im] = {}
                pair_data[test_im]['test_pairs'] = []

            # Wrap pose label with RelaPose, AbsPose objects
            q, t, E_ = im_pairs[(train_im, test_im)]
            ctr, qtr = abs_pose[train_im]
            cte, qte = abs_pose[test_im]
            rela_pose_lbl = RelaPose(q, t)
            train_abs_pose = AbsPose(qtr, ctr, init_proj=True)
            test_abs_pose = AbsPose(qte, cte)
            pair_data[test_im]['test_abs_pose'] = test_abs_pose
            
            # Extract pt correspondences
            pair_id = pair_ids[i]         
            pts, invalid = extract_pair_pts(pair_id, key_points, matches)
            if invalid:
                no_pts_pairs.append(im_pair)
                # TODO: set the accuracy to very large?
                continue
             
            # Estimate essential matrix from pt correspondences and extract relative poses
            K = intrinsic_loader.get_relative_intrinsic_matrix(train_im, test_im)
            p1 = pts[:, 0:2]
            p2 = pts[:, 2:4]
            E, inliers = cv2.findEssentialMat(p1, p2, cameraMatrix=K, method=cv2.FM_RANSAC, threshold=rthres)

            # Wrap ess pair
            (t, R0, R1) = decompose_essential_matrix(E)
            test_pair = EssPair(test_im, train_im, train_abs_pose, rela_pose_lbl, t, R0, R1)
            if test_pair.is_invalid():
                # Invalid pairs that causes 'inf' due to bad retrieval and will corrupt ransac
                continue
                
            pair_data[test_im]['test_pairs'].append(test_pair) 
        total_time = time.time() - start_time
        lprint('Scene:{} Samples total:{} No correspondence pairs:{}. Time total:{:.4f} per_pair:{:.4f}'.format(scene, total_num, len(no_pts_pairs), total_time, total_time / (1.0 * total_num)), log)   
        result_dict[scene] = {}
        result_dict[scene]['pair_data'] = pair_data
        result_dict[scene]['no_pt_pairs'] = no_pts_pairs
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', '-droot',  type=str, default='data/datasets_original/')
    parser.add_argument('--dataset', '-ds',  type=str, default='CambridgeLandmarks',
                        choices=['CambridgeLandmarks', '7Scenes'])
    parser.add_argument('--scenes', '-sc', type=str, nargs='*', default=None)
    parser.add_argument('--colmap_db_root', '-dbdir', type=str, default='data/colmap_dbs')
    parser.add_argument('--db_name', '-db', type=str, default='database.db')
    parser.add_argument('--pair_txt', '-pair', type=str, default='test_pairs.5nn.300cm50m.vlad.minmax.txt')
    parser.add_argument('--train_lbl_txt', type=str, default='dataset_train.txt')
    parser.add_argument('--test_lbl_txt', type=str, default='dataset_test.txt')
    parser.add_argument('--gpu', '-gpu', type=int, default=0)
    parser.add_argument('--cv_ransac_thres', type=float, nargs='*', default=[0.5])
    parser.add_argument('--loc_ransac_thres', type=float, nargs='*', default=[5])
    parser.add_argument('--output_root', '-odir', type=str, default='output/sift/')
    parser.add_argument('--log_txt', '-log', type=str, default='test_results.txt')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    scene_dict = {'CambridgeLandmarks' : ['KingsCollege', 'OldHospital', 'ShopFacade',  'StMarysChurch'],
                  '7Scenes' : ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
                 }
    dataset = args.dataset
    scenes = scene_dict[dataset] if not args.scenes else args.scenes

    out_dir = os.path.join(args.output_root, dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_path = os.path.join(out_dir, args.log_txt)
    log = open(log_path, 'a') 
    lprint('Log results {}'.format(log_path))

    for rthres in args.cv_ransac_thres:
        lprint('>>>>Eval Sift CV Ransac : {} Pair: {}'.format(rthres, args.pair_txt), log)
        ransac_sav_path = os.path.join(out_dir, 'ransacs.rthres{}.npy'.format(rthres))

        # Predict essential matrixes in a structured data dict
        result_dict = predict_essential_matrix(args.data_root, dataset, scenes, 
                                               args.pair_txt, args.train_lbl_txt, args.test_lbl_txt,
                                               args.colmap_db_root, db_name=args.db_name,
                                               rthres=rthres, log=log)
        np.save(os.path.join(out_dir, 'preds.rthres{}.npy'.format(rthres)), result_dict)

        #Global localization via essential-matrix-based ransac                                   
        eval_pipeline_with_ransac(result_dict, log, ransac_thres=args.loc_ransac_thres, 
                                  ransac_iter=10, ransac_miu=1.414, pair_type='ess', 
                                  err_thres=[(0.25, 2), (0.5, 5), (5, 10)], 
                                  save_res_path=ransac_sav_path)
    log.close()
    
if __name__ == '__main__':
    main()