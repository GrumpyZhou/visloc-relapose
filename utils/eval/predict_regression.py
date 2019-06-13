import time
from utils.common.setup_helper import lprint
from utils.pose.localize import *

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
                    (t, R0, R1) = decompose_essential_matrix(E)
                    train_im = batch['im_pair_refs'][0][i]
                    test_pair = EssPair(test_im, train_im, train_abs_pose, rela_pose_lbl, t, R0, R1)
                                        
                if pair_type == 'relapose':
                    # Pose prediction is (t, q)
                    rela_pose_pred = RelaPose(pred_vec[1][i].cpu().data.numpy(), pred_vec[0][i].cpu().data.numpy())
                    test_pair = RelaPosePair(test_im, train_abs_pose, rela_pose_lbl, rela_pose_pred)
                pair_data[test_im]['test_pairs'].append(test_pair) 
        total_time = time.time() - start_time
        lprint('Dataset:{} num_samples:{} total_time:{:.4f} time_per_pair:{:.4f}'.format(dataset, total_num, total_time, total_time / (1.0 * total_num)), log)   
        result_dict[dataset]['pair_data'] = pair_data
    return result_dict