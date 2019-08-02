import argparse
import os
import numpy as np
import cv2
import glob
import time

import torch
import torch.utils.data as data

from utils.common.setup_helper import lprint
from utils.common.ncmatch_config import NCMatchEvalConfig
from networks.ncmatchnet import NCMatchNet
from utils.datasets.camera_intrinsics import get_camera_intrinsic_loader
from utils.datasets.ncmatch_eval import get_test_loaders
from utils.eval.extract_ncmatches import cal_matches
from utils.eval.localize import *

def predict_essential_matrix(data_root, dataset, data_loaders, model,
                             k_size=2, do_softmax=True,
                             rthres=4.0, ncn_thres=0.9, log=None): 
    result_dict = {}
    for scene in data_loaders:
        result_dict[scene] = {}
        data_loader = data_loaders[scene]
        total_num = len(data_loader.dataset)  # Total image pair number
        base_dir = os.path.join(data_root, dataset)
        scene_dir = os.path.join(base_dir, scene)
        intrinsic_loader = get_camera_intrinsic_loader(base_dir, dataset, scene)
  
        # Predict essential matrix over samples
        pair_data = {}
        start_time = time.time()
        for i, batch in enumerate(data_loader): 
            # Load essnet images
            train_im_ref, test_im_ref = batch['im_pair_refs'][0][0], batch['im_pair_refs'][1][0] 
            train_im, test_im = batch['im_pairs']
            train_im = train_im.to(model.device)
            test_im = test_im.to(model.device)

            # Calculate correspondence score map 
            with torch.no_grad():
                # Forward feature to ncn module
                if k_size>1:
                    corr4d,delta4d=model.forward_corr4d(train_im, test_im)
                else:
                    corr4d,delta4d=model.forward_corr4d(train_im, test_im)
                    delta4d=None 

            # Calculate matches
            xA, yA, xB, yB, score = cal_matches(corr4d, delta4d, k_size=k_size,
                                                do_softmax=do_softmax,
                                                matching_both_directions=True,
                                                recenter=False)

            # Scale matches to original pixel level
            w, h = intrinsic_loader.w, intrinsic_loader.h
            matches = np.dstack([xA*w, yA*h, xB*w, yB*h]).squeeze() # N, 4
            K = intrinsic_loader.get_relative_intrinsic_matrix(train_im_ref, test_im_ref)            
            
            # Find essential matrix
            inds = np.where(score > ncn_thres)[0]
            score = score[inds]
            matches = matches[inds, :]
            p1 = matches[:,0:2]
            p2 = matches[:,2:4]
            E, inliers = cv2.findEssentialMat(p1, p2, cameraMatrix=K, method=cv2.FM_RANSAC, threshold=rthres)

            # Dict to saving results in a structured way for later RANSAC
            if test_im_ref not in pair_data:
                pair_data[test_im_ref] = {}
                pair_data[test_im_ref]['test_pairs'] = []
                pair_data[test_im_ref]['cv_inliers'] = []
                
            # For debugging    
            inliers = np.nonzero(inliers)[0]
            inlier_ratio =  len(inliers)/ p1.shape[0]            
            pair_data[test_im_ref]['cv_inliers'].append(inlier_ratio)
            
            # Wrap pose label with RelaPose, AbsPose objects
            rela_pose_lbl = RelaPose(batch['relv_q'][0].data.numpy(), batch['relv_t'][0].data.numpy())
            train_abs_pose = AbsPose(batch['train_abs_q'][0].data.numpy(), batch['train_abs_c'][0].data.numpy(), init_proj=True)
            test_abs_pose = AbsPose(batch['test_abs_q'][0].data.numpy(), batch['test_abs_c'][0].data.numpy())
            pair_data[test_im_ref]['test_abs_pose'] = test_abs_pose

            # Wrap ess pair
            (t, R0, R1) = decompose_essential_matrix(E)
            test_pair = EssPair(test_im_ref, train_im_ref, train_abs_pose, rela_pose_lbl, t, R0, R1)
            if test_pair.is_invalid():
                # Invalid pairs that causes 'inf' due to bad retrieval and will corrupt ransac
                continue
            pair_data[test_im_ref]['test_pairs'].append(test_pair)    
        total_time = time.time() - start_time
        lprint('Scene:{} num_samples:{} total_time:{:.4f} time_per_pair:{:.4f}'.format(scene, total_num, total_time, total_time / (1.0 * total_num)), log)   
        result_dict[scene]['pair_data'] = pair_data
    return result_dict    

def main():
    parser = argparse.ArgumentParser(description='Eval immatch model')
    parser.add_argument('--data_root', '-droot',  type=str, default='data/datasets_original/')
    parser.add_argument('--dataset', '-ds',  type=str, default='CambridgeLandmarks',
                        choices=['CambridgeLandmarks', '7Scenes'])
    parser.add_argument('--scenes', '-sc', type=str, nargs='*', default=None)
    parser.add_argument('--image_size', '-imsize', type=int, default=None)
    parser.add_argument('--scale_pts', action='store_true')
    parser.add_argument('--pair_txt', '-pair', type=str, default='test_pairs.5nn.300cm50m.vlad.minmax.txt')
    parser.add_argument('--ckpt_dir', '-cdir', type=str, default=None)
    parser.add_argument('--ckpt_name', '-ckpt', type=str, default=None)
    parser.add_argument('--feat', '-feat', type=str, default=None)
    parser.add_argument('--ncn', '-ncn', type=str, default=None)
    parser.add_argument('--gpu', '-gpu', type=int, default=0)
    parser.add_argument('--cv_ransac_thres', type=float, nargs='*', default=[4.0])
    parser.add_argument('--loc_ransac_thres', type=float, nargs='*', default=[15])
    parser.add_argument('--ncn_thres', type=float, default=0.9)
    parser.add_argument('--posfix', type=str, default='imagenet+ncn')
    parser.add_argument('--out_dir', '-o', type=str, default='output/ncmatch_5pt/loc_results')  

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Data Loading
    data_loaders = get_test_loaders(image_size=args.image_size, 
                                    dataset=args.dataset, 
                                    scenes=args.scenes,
                                    pair_txt=args.pair_txt,
                                    data_root=args.data_root)

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, '{}.txt'.format(args.posfix))
    log = open(out_path, 'a') 
    lprint('Log results {} posfix: {}'.format(out_path, args.posfix))

    if args.ckpt_dir is None:
        ckpts = [None]
    elif args.ckpt_name:
        # Load the target checkpoint
        ckpts = [os.path.join(args.ckpt_dir, args.ckpt_name)]
    else:
        # Load all checkpoints under ckpt_dir
        ckpts = glob.glob(os.path.join(args.ckpt_dir, 'checkpoint*')) 

    print('Start evaluation, ckpt to eval: {}'.format(len(ckpts)))
    for ckpt in ckpts:
        # Load models
        lprint('\n\n>>>Eval ImMatchNet:pair_txt: {}\nckpt {} \nfeat {} \nncn {}'.format(args.pair_txt, ckpt, args.feat, args.ncn), log)
        config = NCMatchEvalConfig(weights_dir=ckpt, feat_weights=args.feat,
                                   ncn_weights=args.ncn, early_feat=True)
        matchnet = NCMatchNet(config)

        for rthres in args.cv_ransac_thres:
            lprint('\n>>>>cv_ansac : {} ncn_thres: {}'.format(rthres, args.ncn_thres), log)
            result_dict = predict_essential_matrix(args.data_root, args.dataset, data_loaders, 
                                                   matchnet, do_softmax=True,
                                                   rthres=rthres, ncn_thres=args.ncn_thres, log=log)

            np.save(os.path.join(out_dir, 'preds.cvr{}_ncn{}.{}.npy'.format(rthres, args.ncn_thres, args.posfix)), result_dict)
            eval_pipeline_with_ransac(result_dict, log, ransac_thres=args.loc_ransac_thres, 
                                      ransac_iter=10, ransac_miu=1.414, pair_type='ess', 
                                      err_thres=[(0.25, 2), (0.5, 5), (5, 10)], 
                                      save_res_path=None)  
    log.close()

if __name__ == '__main__':
    main()    