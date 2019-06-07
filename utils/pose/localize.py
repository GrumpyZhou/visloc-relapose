import os
import torch
import numpy as np
import itertools
import time
from transforms3d.quaternions import quat2mat, mat2quat
from utils.common.setup_helper import lprint
from utils.pose.transform import angs2pose
from utils.pose.measure import cal_quat_angle_error, cal_vec_angle_error
import warnings

def eval_prediction(data_loaders, net, log, pair_type='ess'):
    '''The function evaluates network predictions and prepare results for RANSAC and convenient accuracy calculation.
    Args:
        - data_loaders: a dict of data loaders(torch.utils.data.DataLoader) for all datasets.
        - net: the network model to be tested.
        - model_type: specifies the type network prediction. Options: 'ess' or 'relapose'.
        - log: path of the txt file for logging
    Return:
       A data dictionary that is designed especially to involve all necessary relative and absolute pose information required
       by the RANSAC algorithm and error metric accuracy calculations, since the precalculation can drastically improve the running time. 
       Basically, a key in the data dict is a test_im and the corresponding value is a sub dict. The sub dict involves:
       - key:'test_abs_pos', value: the absolute pose label of the test_im (for absolute pose accuracy calculation)
       - key:'test_pairs', value: a list of pair data linked to the test_im. The pair data are encapsulated inside either
       a RelaPosePair object or an EssPair object depending on the type of regression results.
    '''   
    result_dict = {}
    for dataset in data_loaders:
        start_time = time.time()
        data_loader = data_loaders[dataset]
        total_num = len(data_loaders[dataset].dataset)
        pair_data = {}
        result_dict[dataset] = {}
        ess_predicts = []  #TOREMOVE
        for i, batch in enumerate(data_loader):
            pred_vec = net.predict_(batch)

            # Calculate poses per query image
            for i, test_im in enumerate(batch['im_pair_refs'][1]):
                if test_im not in pair_data:
                    pair_data[test_im] = {}
                    pair_data[test_im]['test_pairs'] = []

                # Wrap pose label with RelaPose, AbsPose objects
                rela_pose_lbl = RelaPose(batch['relv_q'][i].data.numpy(), batch['relv_t'][i].data.numpy())
                train_abs_pose = AbsPose(batch['train_abs_q'][i].data.numpy(), batch['train_abs_c'][i].data.numpy(), init_proj=True)
                test_abs_pose = AbsPose(batch['test_abs_q'][i].data.numpy(), batch['test_abs_c'][i].data.numpy())
                pair_data[test_im]['test_abs_pose'] = test_abs_pose

                # Estimate relative pose
                if pair_type == 'ess':
                    # Pose prediction are essential matrix
                    E = pred_vec[i].cpu().data.numpy().reshape((3,3))
                    ess_predicts.append(E) #TOREMOVE
                    (t, R0, R1) = decompose_essential_matrix(E)
                    train_im = batch['im_pair_refs'][0][i]
                    test_pair = EssPair(test_im, train_im, train_abs_pose, rela_pose_lbl, t, R0, R1)
                    
                if pair_type == 'angess':
                    # Pose prediction are 5 angles
                    theta, phi, ai, aj, ak = pred_vec[i].cpu().data.numpy()
                    t, q = angs2pose(theta, phi, ai, aj, ak)
                    rela_pose_pred = RelaPose(q, t)
                    test_pair = RelaPosePair(test_im, train_abs_pose, rela_pose_lbl, rela_pose_pred)
                    
                if pair_type == 'relapose':
                    # Pose prediction is (t, q)
                    rela_pose_pred = RelaPose(pred_vec[1][i].cpu().data.numpy(), pred_vec[0][i].cpu().data.numpy())
                    test_pair = RelaPosePair(test_im, train_abs_pose, rela_pose_lbl, rela_pose_pred)
                pair_data[test_im]['test_pairs'].append(test_pair) 
        total_time = time.time() - start_time
        lprint('Dataset:{} num_samples:{} total_time:{:.4f} time_per_pair:{:.4f}'.format(dataset, total_num, total_time, total_time / (1.0 * total_num)), log)   
        result_dict[dataset]['pair_data'] = pair_data
        result_dict[dataset]['ess_preds'] = ess_predicts #TOREMOVE
    return result_dict

def eval_pipeline_with_ransac(result_dict, log, ransac_thres, ransac_iter, 
                              ransac_miu, pair_type, err_thres, save_res_path=None):
    lprint('>>>>Evaluate model with Ransac(iter={}, miu={}) Error thres:{})'.format(ransac_iter, ransac_miu, err_thres), log)
    t1 = time.time()
    best_abs_err = None # TODO: not used for now, remove it in the end
    for thres in ransac_thres:
        avg_err = []
        avg_pass = []
        lprint('\n>>Ransac threshold:{}'.format(thres), log)
        loc_results_dict = {}
        for dataset in result_dict:
            start_time = time.time()
            pair_data = result_dict[dataset]['pair_data']
            loc_results_dict[dataset] = {} if save_res_path else None
            if pair_type == 'angess':    # Since angles have been converted to relative poses
                pair_type = 'relapose'
            tested_num, approx_queries, pass_rate, err_res = ransac(pair_data, thres, in_iter=ransac_iter, 
                                                                    pair_type=pair_type, err_thres=err_thres,
                                                                    loc_results=loc_results_dict[dataset])
            avg_err.append(err_res)
            avg_pass.append(pass_rate)
            total_time = time.time() - start_time
            dataset_pr_len = min(10, len(dataset))
            lprint('Dataset:{dataset} Bad/All:{approx_num}/{tested_num}, Rela:({err_res[0]:.2f}deg,{err_res[1]:.2f}deg) Abs:({err_res[2]:.2f}m,{err_res[4]:.2f}deg)/{err_res[3]:.2f}deg) Pass:'.format(dataset=dataset[0:dataset_pr_len], 
           approx_num=len(approx_queries), tested_num=tested_num, err_res=err_res) + '/'.join('{:.2f}%'.format(v) for v in pass_rate), log)
          
        avg_err = tuple(np.mean(avg_err, axis=0))
        avg_pass = tuple(np.mean(avg_pass, axis=0)) if len(err_thres) > 1 else tuple(avg_pass)
        if best_abs_err is not None:
            if best_abs_err[0] < avg_err[2]:
                best_abs_err = (avg_err[2], avg_err[4])
        else:
            best_abs_err = (avg_err[2], avg_err[4])
        lprint('Avg: Rela:({err_res[0]:.2f}deg,{err_res[1]:.2f}deg) Abs:({err_res[2]:.2f}m,{err_res[4]:.2f}deg;{err_res[3]:.2f}deg) Pass:'.format(err_res=avg_err) + '/'.join('{:.2f}%'.format(v) for v in avg_pass), log)

        if save_res_path:
            np.save(save_res_path, loc_results_dict)
    time_stamp = 'Ransac testing time: {}s\n'.format(time.time() - t1)
    lprint(time_stamp, log)
    return best_abs_err

def eval_pipeline_without_ransac(result_dict, err_thres=(2, 5), log=None):
    avg_rela_t_err = []         # Averge relative translation error in angle over datasets
    avg_rela_q_err = []         # Average relative roataion(quternion) error in angle over datasets
    avg_abs_c_dist_err = []     # Averge absolute position error in meter over datasets
    avg_abs_c_ang_err = []      # Averge absolute position error in angle over datasets
    avg_abs_q_err = []          # Averge absolute roataion(quternion) angle error over dataset
    
    for dataset in result_dict:
        pair_data = result_dict[dataset]['pair_data']
        lprint('>>Testing dataset: {}, testing samples: {}'.format(dataset, len(pair_data)), log)
        
        # Calculate relative pose error
        rela_t_err, rela_q_err = cal_rela_pose_err(pair_data)
        avg_rela_t_err.append(rela_t_err)
        avg_rela_q_err.append(rela_q_err)
        
        # Calculate testing pose error with all training images
        abs_c_dist_err, abs_c_ang_err, abs_q_err, passed = cal_abs_pose_err(pair_data, err_thres)
        avg_abs_c_dist_err.append(abs_c_dist_err)
        avg_abs_c_ang_err.append(abs_c_ang_err)                    
        avg_abs_q_err.append(abs_q_err)
                 
        lprint('rela_err ({:.2f}deg, {:.2f}deg) abs err: ({:.2f}m, {:.2f}deg; {:.2f}deg, pass: {:.2f}%)'.format(rela_t_err, rela_q_err, abs_c_dist_err, abs_c_ang_err, abs_q_err, passed), log)
    eval_val = (np.mean(avg_rela_t_err), np.mean(avg_rela_q_err), np.mean(avg_abs_c_dist_err), np.mean(avg_abs_c_ang_err), np.mean(avg_abs_q_err))
    lprint('>>avg_rela_err ({eval_val[0]:.2f} deg, {eval_val[1]:.2f} deg) avg_abs_err ({eval_val[2]:.2f} m, {eval_val[3]:.2f}deg; {eval_val[4]:.2f}deg)'.format(eval_val=eval_val), log)
    return eval_val

def cal_rela_pose_err(pair_data):
    """Calculate relative pose median errors directly over all tested pairs, including:
       - relative translation angle error
       - relative quaternion angle error
    """
    rela_q_err = []
    rela_t_err = []
    for test_im in pair_data:
        test_pair_list = pair_data[test_im]['test_pairs']
        for test_pair in test_pair_list:
            rela_t_err.append(cal_vec_angle_error(test_pair.rela_pose_pred.t, test_pair.rela_pose_lbl.t)) 
            rela_q_err.append(cal_quat_angle_error(test_pair.rela_pose_pred.q, test_pair.rela_pose_lbl.q))
    return np.median(rela_t_err), np.median(rela_q_err)

def cal_abs_pose_err(pair_data, err_thres=(2, 5)):
    """Calculate absolute pose median errors directly (No RANSAC) over all tested pairs, including:
       - absolute positional distance error in meter
       - absolute positional angle error
       - absolute rotational angle error
    """
    abs_c_dist_err = []
    abs_c_ang_err = []
    abs_q_err = []
    passed = 0
    for test_im in pair_data:
        test_abs_pose = pair_data[test_im]['test_abs_pose']
        test_pair_list = pair_data[test_im]['test_pairs'] 
        k = len(test_pair_list)
        if k == 1:
            continue    # Triangulation requires at least 2 training images

        # Estimate absolute pose of test image
        abs_q_pred_list = []  
        correspondence = []
        train_abs_c_list = []
        for test_pair in test_pair_list:
            correspondence.append((test_pair.x_te, test_pair.train_abs_pose.p))
            abs_q_pred_list.append(test_pair.abs_q_pred)
            train_abs_c_list.append(test_pair.train_abs_pose.c)
        abs_c_pred = triangulate_multi_views(correspondence)
        cerr = np.linalg.norm(test_abs_pose.c - abs_c_pred)
        abs_c_dist_err.append(cerr)
        train_abs_c = np.vstack(train_abs_c_list)
        abs_c_ang_err.append(np.mean(cal_vec_angle_error(test_abs_pose.c - train_abs_c, abs_c_pred - train_abs_c)))     # Angle between abs_pose label and prediction, with train image as reference point
        abs_q_pred = np.mean(np.vstack(abs_q_pred_list), axis=0)  
        qerr = cal_quat_angle_error(test_abs_pose.q, abs_q_pred)
        abs_q_err.append(qerr)
        
        if cerr < err_thres[0] and qerr < err_thres[1]:
            passed += 1
                
    return np.median(abs_c_dist_err), np.median(abs_c_ang_err), np.median(abs_q_err), 100.0*passed/len(abs_q_err)


########################
#### RANSAC METHODS#####
########################
def ransac(pair_data, inlier_thres, thres_multiplier=1.414, in_iter=10, pair_type='ess', 
           err_thres=[(0.25, 2), (0.5, 5), (5, 10)], loc_results=None):
    abs_c_dist_err = []
    abs_c_ang_err = []
    abs_q_err = []
    rela_t_err = []
    rela_q_err = []
    passed = [0 for thres in err_thres]
    approx_queries = []
    for test_im in pair_data:      
        test_abs_pose = pair_data[test_im]['test_abs_pose']
        test_pair_list = pair_data[test_im]['test_pairs'] 
        num_pair = len(test_pair_list)
        
        if num_pair == 0:
            # There's no pair predictes valid essentials
            # Manually set big errors, median error should be robust to such outliers
            c_err = 1000
            q_err = 180
            abs_c_dist_err.append(c_err)
            abs_c_ang_err.append(q_err)
            abs_q_err.append(q_err)  
            rela_t_err.append(q_err)
            rela_q_err.append(q_err)
        else:
            # Run Ransac algorithm
            inlier_best = []
            abs_pose_best = None
            approximated = False
            inlier_min_samples = list(itertools.combinations(range(num_pair), 2)) # Check all possible pairs as minimal inlier samples 
            for inlier_min in inlier_min_samples:
                # The initial hypothesis in different manner according to the structure(type) of test pair
                if pair_type == 'ess':
                    pair0, pair1 = test_pair_list[inlier_min[0]], test_pair_list[inlier_min[1]]
                    err_min = 1000
                    id0, id1 = -1, -1

                    # Determine the two rotations for the pair by choosing the combination with smallest angle error
                    for i in range(2):
                        for j in range(2):
                            err = cal_quat_angle_error(pair0.abs_q_pred[i], pair1.abs_q_pred[j])
                            if err < err_min:
                                err_min = err
                                id0, id1 = i, j
                    # Use the average quaternion over 2 pairs
                    abs_q_hypo = np.mean(np.vstack([pair0.abs_q_pred[id0], pair1.abs_q_pred[id1]]), axis=0)
                    x0, x1 = pair0.x_te[id0], pair1.x_te[id1]
                    p0, p1 = pair0.train_abs_pose.p, pair1.train_abs_pose.p
                    abs_c_hypo = triangulate_two_views(x0, p0, x1, p1)
                    abs_pose_hypo = AbsPose(abs_q_hypo, abs_c_hypo) 
                if pair_type == 'relapose':
                    abs_pose_hypo = estimate_model(test_pair_list, inlier_min, pair_type)
                inlier_hypo = find_inliers(abs_pose_hypo, test_pair_list, inlier_thres, pair_type=pair_type) 

                # Perform local optimation step if this hypo has so far most inliers
                if len(inlier_hypo) >= 2 and len(inlier_hypo) > len(inlier_best):
                    inlier_best = inlier_hypo
                    abs_pose_best = abs_pose_hypo

                    # local optimisation
                    inlier_local_best, pose_local_best = local_optimisation(test_pair_list, abs_pose_best, 
                                                                            thres_multiplier, inlier_thres, 
                                                                            in_iter, pair_type)
                    if len(inlier_local_best) > len(inlier_best):
                        inlier_best = inlier_local_best
                        abs_pose_best = pose_local_best
                        

            if abs_pose_best is None or len(inlier_best) == 0:
                # In this case, either the pair has bad confidence 
                # Or there's one training pair for this query
                # Use pose of training to approximate the pose
                inlier_id = 0
                pair = test_pair_list[inlier_id]
                abs_pose_best = pair.train_abs_pose
                inlier_best = [inlier_id]
                approx_queries.append(test_im)
                approximated = True
                
            if pair_type == 'ess':
                # Identify the correct relative translation based on the best hypothesis
                find_inliers(abs_pose_best, test_pair_list, inlier_thres, 
                             pair_type=pair_type, update_trans=True)  # Identify t and update

            # Save results for further analysis
            if loc_results is not None: # TODO: Separate process for error analysis
                res = {}
                res['abs_pose_lbl'] = test_abs_pose
                res['abs_pose_pred'] = abs_pose_best
                res['relv_pose_list'] = test_pair_list
                res['inliers'] = inlier_best
                res['approximated'] = approximated
                loc_results[test_im] = res

            # Calculate relative error with best inlier set
            train_abs_c_list = []
            t_err = []
            q_err = []
            for i in inlier_best:
                test_pair = test_pair_list[i]
                train_abs_c_list.append(test_pair.train_abs_pose.c)
                if pair_type == 'ess':
                    t_err.append(cal_vec_angle_error(test_pair.t, test_pair.rela_pose_lbl.t)) 
                    q_err.append(cal_quat_angle_error(test_pair.get_rela_q(), test_pair.rela_pose_lbl.q))
                if pair_type == 'relapose':
                    t_err.append(cal_vec_angle_error(test_pair.rela_pose_pred.t, test_pair.rela_pose_lbl.t)) 
                    q_err.append(cal_quat_angle_error(test_pair.rela_pose_pred.q, test_pair.rela_pose_lbl.q))
            rela_t_err.append(np.mean(t_err))
            rela_q_err.append(np.mean(q_err))
            
            # Calculate absolute error
            if len(train_abs_c_list) > 1:
                train_abs_c = np.vstack(train_abs_c_list) 
            else:
                train_abs_c = train_abs_c_list[0]
                train_abs_c.reshape((1, 3))
            abs_pose_pred = abs_pose_best 
            cerr = np.linalg.norm(test_abs_pose.c - abs_pose_pred.c)
            abs_c_dist_err.append(cerr)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    if approximated:
                        # Prevent zero division
                        abs_c_ang_err.append(0.0)
                    else:
                        # Angle between abs_pose label and prediction, with train image as reference point
                        abs_c_ang_err.append(np.mean(cal_vec_angle_error(test_abs_pose.c - train_abs_c, abs_pose_pred.c - train_abs_c)))
                except Warning: 
                    print('Warning catched during abs angle error calculation')
                    print('Test im {}, num_pair {}'.format(test_im, len(test_pair_list)))

            qerr = cal_quat_angle_error(test_abs_pose.q, abs_pose_pred.q).squeeze()
            abs_q_err.append(qerr)   
         
        # DSAC eval criterion: cerr < thres (m) & qerr < thres (deg)
        for i, thres in enumerate(err_thres):
            cerr_thres, qerr_thres = thres
            if cerr < cerr_thres and qerr < qerr_thres:
                passed[i] += 1
    num_tested = len(abs_c_dist_err)
    pass_rate = [100.0 * count / num_tested for count in passed]
    return num_tested, approx_queries, pass_rate, (np.median(rela_t_err), np.median(rela_q_err), np.median(abs_c_dist_err), 
           np.median(abs_c_ang_err), np.median(abs_q_err))

def local_optimisation(test_pair_list, abs_pose_best, thres_multiplier, thres, in_iter, pair_type):
    # Re-evaluate model and inliers with threshold multiplied by thres_multiplier
    inlier_mult = find_inliers(abs_pose_best, test_pair_list, thres_multiplier*thres, pair_type=pair_type)
    abs_pose_mult = estimate_model(test_pair_list, inlier_mult, pair_type)
    inlier_base = find_inliers(abs_pose_mult, test_pair_list, thres, pair_type=pair_type) 
 
    # Evaluate model from subsample of inlier_base
    inlier_base_sample = list(inlier_base)
    all_abs_poses = [abs_pose_best, abs_pose_mult]
    num_inlier_subsample = min(14, int(len(inlier_base)/2))
    if num_inlier_subsample > 2:
        for i in range(in_iter):
            np.random.shuffle(inlier_base_sample)
            inlier_subsample = inlier_base_sample[:num_inlier_subsample]
            abs_pose_subsample = estimate_model(test_pair_list, inlier_subsample, pair_type)
            all_abs_poses.append(abs_pose_subsample)
    
    # Identify the best model
    inlier_local_best = []
    pose_local_best = None
    for abs_pose in all_abs_poses:
        inlier_ = find_inliers(abs_pose, test_pair_list, thres, pair_type=pair_type) 
        if len(inlier_) > len(inlier_local_best):
            inlier_local_best = inlier_
            pose_local_best = abs_pose
    return inlier_local_best, pose_local_best

def find_inliers(hypo_abs_pose, test_pair_list, thres, pair_type='ess', update_trans=False):
    """
    Find inliers from the full sample set based on the hypothesised absolute pose of the test image.
    The train image is counted as an inlier if the estimated translation angle error between 
    it and the test image is within the threshold.
    Args:
        - hypo_abs_pose: AbasPose object represents the absolute pose hypothesis of the test image
        - test_pair_list: the full set of testing pairs, i.e., a list of RelaPosePair/EssPair objects
        - thres: error threshold to filter the outliers
        - pair_type: specifies the type pair objects: 'relapose'->RelaPosePair, 'ess'->EssPair
        - update_trans: specifies whether to determine the correct sign of the relative translation, 
          this is only used when the pair_type is 'ess'. And the update is performed after the best absolute hypothese
          is found and the sign giving smaller angle difference from the hypothesed translation is picked.
    Return:
        - the inlier indices of test pairs from the test_pair_list
    """
    inliers = []
    k = len(test_pair_list)
    for i in range(k):
        test_pair = test_pair_list[i]
        
        # Estimate relative translation (from query to train)based on the hypothesis
        train_abs_pose = test_pair.train_abs_pose        
        rela_t_est = train_abs_pose.r.dot(hypo_abs_pose.c - train_abs_pose.c)  

        # Optimal reltaive pose predicted by the network
        if pair_type == 'ess':
            # Identify the correct rotation of each pair with hypothesis
            err0 = cal_quat_angle_error(hypo_abs_pose.q, test_pair.abs_q_pred[0]) 
            err1 = cal_quat_angle_error(hypo_abs_pose.q, test_pair.abs_q_pred[1])
            rid = np.argmin([err0, err1])
            test_pair.set_rid(rid)
            rela_r_opt = test_pair.R[rid]
            rela_t_opt = test_pair.t
        if pair_type == 'relapose':
            rela_r_opt = test_pair.rela_pose_pred.r
            rela_t_opt = test_pair.rela_pose_pred.t
        t_est = rela_t_est
        t_opt = - rela_r_opt.T.dot(rela_t_opt)  # reversed direction, i.e., from query to train im
        
        err = np.inf
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                if np.linalg.norm(t_est) == 0.0:
                    # Training and testing image has same position
                    err = 0.0  # Check whether this is appropriate
                else:
                    # Calculate translation angle error and locate inliers by threshoding
                    err = cal_vec_angle_error(t_est, t_opt)
                    if pair_type == 'ess':
                        # Identify the correct translation that giving smaller error
                        t_opt_ = - t_opt
                        err_ = cal_vec_angle_error(t_est, t_opt_)
                        if err_ < err:
                            err = err_
                            if update_trans:
                                test_pair.set_opposite_trans_pred()  # Update to the correct translation in the pair data   
            except Warning: 
                print('Warning catched during find inlier calculation')
                print('Test im {}, Train {}'.format(test_pair.test_im, test_pair.train_im))
                
        if err < thres:
            inliers.append(i)
    return inliers

def estimate_model(test_pair_list, inliers, pair_type):
    """Estimate absolute pose of test image 
    Args:
        - test_pair_list: the full set of testing pairs, i.e., a list of RelaPosePair/EssPair objects
        - inliers: list of indices pointing to the inlier test pairs
        - pair_type: specifies the type pair objects: 'relapose'->RelaPosePair, 'ess'->EssPair
    Return:
        - an AbasPose object representing the absolute pose prediction with the given inliers
    """
    abs_q_pred_list = []  
    correspondence = []
    for i in inliers:
        test_pair = test_pair_list[i]
        if pair_type == 'ess':
            rid = test_pair.rid
            correspondence.append((test_pair.x_te[rid], test_pair.train_abs_pose.p))
            abs_q_pred_list.append(test_pair.abs_q_pred[rid])        
        if pair_type == 'relapose':
            correspondence.append((test_pair.x_te, test_pair.train_abs_pose.p))
            abs_q_pred_list.append(test_pair.abs_q_pred)
    abs_c_pred = triangulate_multi_views(correspondence)
    abs_q_pred = np.mean(np.vstack(abs_q_pred_list), axis=0) 
    return AbsPose(abs_q_pred, abs_c_pred)

#############################
####EPIPOLAR CALCULATION#####
#############################
def triangulate_two_views(x1, p1, x2, p2):
    """Triangulate a 3d point from 2 views 
    Args:
        - x1: point correspondence of target 3d point in 1st view
        - p1: projection matrix of 1st view i.e. [R1|t1]
        - x2: point correspondence  of target 3d point in 2nd view
        - p1: projection matrix of 2nd view i.e. [R2|t2]
    Return:
        - X: triangulated 3d point, shape (3,)
    """
    A_rows = []
    A_rows.append(np.expand_dims(x1[0]*p1[2,:] - p1[0,:], axis=0))
    A_rows.append(np.expand_dims(x1[1]*p1[2,:] - p1[1,:], axis=0))
    A_rows.append(np.expand_dims(x2[0]*p2[2,:] - p2[0,:], axis=0))
    A_rows.append(np.expand_dims(x2[1]*p2[2,:] - p2[1,:], axis=0))
    A = np.concatenate(A_rows, axis=0)

    # Find null space of A
    u, s, vh = np.linalg.svd(A)
    X = vh[-1,:]    # Last column of v
    X = X[:3] / X[3]
    return X

def triangulate_multi_views(correspondence):
    """Triangulate a 3d point from multiple views
    Args:
        - correspondence = list of (xi, pi) where
            xi: 2d point correspondence of target 3d point in i-th view
            pi: projection matrix of i-th view i.e. [Ri|ti]
    Return:
        - X: triangulated 3d point, shape (3,)
    """
    A_rows = []
    for (xi, pi) in correspondence:
        A_rows.append(np.expand_dims(xi[0]*pi[2,:] - pi[0,:], axis=0))
        A_rows.append(np.expand_dims(xi[1]*pi[2,:] - pi[1,:], axis=0))
    A = np.concatenate(A_rows, axis=0)

    # Find null space of A
    u, s, vh = np.linalg.svd(A)
    X = vh[-1,:]    # Last column of v
    X = X[:3] / X[3]
    return X

def compose_projection_matrix(R, t):
    """Construct projection matrix 
    Args:
        - R: rotation matrix, size (3,3);
        - t: translation vector, size (3,);
    Return:
        - projection matrix [R|t], size (3,4)
    """
    return np.hstack([R, np.expand_dims(t, axis=1)])

def hat(vec):
    """Skew operator
    Args:
        - vec: vector of size (3,) to be transformed;
    Return: 
        - skew-symmetric matrix of size (3, 3)
    """ 
    [a1, a2, a3] = list(vec)
    skew = np.array([[0, -a3, a2],[a3, 0, -a1],[-a2, a1, 0]])
    return skew

def project_onto_essential_space(F):
    u, s, vh = np.linalg.svd(F)
    a = (s[0] + s[1]) / 2
    s_ = np.array([a, a, 0])
    E = u.dot(np.diag(s_)).dot(vh)
    return E

def essential_matrix_from_pose(R, t):
    """Calculate essential matrix
    Args:
        - R: rotation matrix, size (3,3);
        - t: translation vector, size (3,);
    Return:
        - essential matrix, size (3,3)
    """
    t = t/np.linalg.norm(t)     # force translation to be unit length
    t_skew = hat(t)
    E = t_skew.dot(R)
    return E.astype(np.float32)

def decompose_essential_matrix(E):
    """Extract possible pose from essential matrix
    Args:
        - E: essential matrix, shape(3,3)
    Return:
        - t: one possible translation, the other is -t
        - R1, R2: possible rotation
    """
    u, s, vh = np.linalg.svd(E)
    if np.linalg.det(u) < 0 or np.linalg.det(vh) < 0:
        u,s,vh = np.linalg.svd(-E)
    t = u[:,2]
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = u.dot(w).dot(vh)
    R2 = u.dot(w.T).dot(vh)
    return (t,R1,R2)

#############################
####DATA WRAPPER CLASSES#####
#############################
class AbsPose:
    def __init__(self, q, c, init_proj=False):
        """Define an absolute camera pose with respect to the global coordinates
        Args:
            - init_proj: whether initialize projection matrix for this instance
        Attributes:
            - c: absolute position of the camera in global system, shape (3,)
            - q: absolute orientation (in quaternion) of the camera in global system, shape (4,)
            - r: absolute orientation (in rotation) of the camera in global system, shape (3, 3)
            - t: translation vector of the pose 
            - p: the projection matrix that transforms a point in global coordinates to this camera's local coordinate
        """
        self.q = q
        self.r = quat2mat(self.q)
        self.c = c
        self.t = -self.r.dot(self.c)
        if init_proj:
            self.p = compose_projection_matrix(self.r, self.t)
    
class RelaPose:
    def __init__(self, q, t):
        """Define a relaitve camera pose
        Attributes:
            - q: relative orientation (in quaternion), shape (4,)
            - r: relative orientation (in rotation), shape (3, 3)
            - t: relative translation vector of the pose 
        """
        self.q = q
        self.r = quat2mat(self.q)
        self.t = t

class RelaPosePair:
    '''This class structures necessary information related to a testing pair for the relative pose regression models'''
    def __init__(self, test_im, train_abs_pose, rela_pose_lbl, rela_pose_pred):
        """Initialize the relative pose data information
        Attributes:
            - test_im: string, the name of the test_im
            - train_abs_pose: AbsPose object, the absolute pose ground truth of the train_im 
            - rela_pose_lbl: RelaPose object, relative pose ground truth        
            - rela_pose_pred: RelaPose object, predicted relative pose by the network
            - x_te : the 2d point correspondence of test_im in this train_im 
            - abs_r_pred : the predicted absolute rotation (SO(3) matrix) of test_im by this train_im		
            - abs_q_pred : the predicted absolute rotation in quaternion
        """
        self.test_im = test_im
        self.train_abs_pose = train_abs_pose
        self.rela_pose_lbl = rela_pose_lbl
        self.rela_pose_pred = rela_pose_pred
        x_te = - self.rela_pose_pred.r.T.dot(self.rela_pose_pred.t) 
        self.x_te = x_te[:2] / x_te[2]
        self.abs_r_pred = self.rela_pose_pred.r.dot(self.train_abs_pose.r)
        self.abs_q_pred = mat2quat(self.abs_r_pred)
    

class EssPair:
    '''This class structures necessary information related to a testing pair for the essential matrix regression models'''
    def __init__(self, test_im, train_im, train_abs_pose, rela_pose_lbl, t, R0, R1):
        """Initialize the relative pose data information
        Attributes:
            - test_im: string, the name of the test_im
            - train_abs_pose: AbsPose object, the absolute pose ground truth of the train_im 
            - rela_pose_lbl: RelaPose object, relative pose ground truth        
            - t: relative translation extracted from an essential matrix, undetermined up to a sign at intialize time. 
                 The sign will be identified in RANSAC using set_opposite_trans_pred()
            - R0, R1: two possible rotation matrices extracted from an essential matrix
            - rid: the index of the correct rotation, either 0 or 1. The rid is set during RANSAC with set_rid()
        The followings are calculated correspondingly using R0 and R1
            - x_te : the two possible 2d point correspondences of test_im in this train_im  
            - abs_r_pred : the two possible predicted absolute rotations in SO(3) matrices of test_im by this train_im		
            - abs_q_pred : the two possible predicted absolute rotations in quaternion
        """
        self.train_im = train_im
        self.test_im = test_im
        self.train_abs_pose = train_abs_pose
        self.rela_pose_lbl = rela_pose_lbl
        self.rela_pose_pred = None
        self.t = t
        self.R = [R0, R1]
        self.abs_r_pred = []
        self.abs_q_pred = []
        self.x_te = []
        for i in range(2):
            R = self.R[i]
            x_te = - R.T.dot(self.t) 
            if x_te[2] == 0:
                self.x_te.append(np.array([np.inf, np.inf]))
            else:
                self.x_te.append(x_te[:2] / x_te[2])
            self.abs_r_pred.append(R.dot(self.train_abs_pose.r))   # r_t1 = R*r_tr
            self.abs_q_pred.append(mat2quat(self.abs_r_pred[i]))

    def set_rid(self, rid):
        self.rid = rid

    def set_opposite_trans_pred(self):
        self.t = - self.t

    def get_rela_q(self):
        return mat2quat(self.R[self.rid])
    
    def is_invalid(self):
        return np.any(np.isinf(self.x_te))
