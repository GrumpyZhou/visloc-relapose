import argparse
import os
import time
import datetime
import torch
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import networks

from utils.common.setup_helper import RelaPoseConfig, lprint, make_deterministic, config2str
from utils.datasets.relapose import VisualLandmarkDataset, get_datasets
from utils.common.visdom import RelaPoseTmp
from utils.pose.localize import eval_prediction, eval_pipeline_with_ransac

def train(net, config, log, train_loader, val_loaders=None):
    # Setup visdom monitor
    visdom = RelaPoseTmp(legend_tag=config.optim_tag,
                         viswin=config.viswin, visenv=config.visenv,
                         vishost=config.vishost, visport=config.visport)
    loss_meter, pos_acc_meter, rot_acc_meter = visdom.get_meters()
    
    print('Start training from {config.start_epoch} to {config.epochs}.'.format(config=config))
    start_time = time.time()
    for epoch in range(config.start_epoch, config.epochs):
        net.train() # Switch to training mode
        loss = net.train_epoch(train_loader, epoch)
        if (epoch+1) % 2 == 0:
            lprint('Epoch {}, loss:{}'.format(epoch+1, loss), log)
            loss_meter.update(X=epoch+1, Y=loss) # Update loss meter
                
        # Always save current training results to prevent lost training
        current_ckpt ={'last_epoch': epoch,
                       'network': config.network,
                       'state_dict': net.state_dict(),
                       'optimizer' : net.optimizer.state_dict()}
        torch.save(current_ckpt, os.path.join(config.ckpt_dir, 'checkpoint.current{}.pth'.format(epoch+1)))
        if config.validate and (epoch+1) % config.validate == 0 and epoch > 0 :
            # Evaluate on validation set
            abs_err = test(net, config, log, val_loaders)
            ckpt ={'last_epoch': epoch,
                   'network': config.network,
                   'state_dict': net.state_dict(),
                   'optimizer' : net.optimizer.state_dict()}
            ckpt_name = 'checkpoint_{epoch}_{abs_err[0]:.2f}m_{abs_err[1]:.2f}deg.pth'.format(epoch=(epoch+1), abs_err=abs_err)
            torch.save(ckpt, os.path.join(config.ckpt_dir, ckpt_name))
            lprint('Save checkpoint: {}'.format(ckpt_name), log)

            # Update validation acc
            pos_acc_meter.update(X=epoch+1, Y=abs_err[0])
            rot_acc_meter.update(X=epoch+1, Y=abs_err[1])
        visdom.save_state()
    lprint('Total training time {0:.4f}s'.format((time.time() - start_time)), log)

def test(net, config, log, test_loaders, sav_res_name=None, 
         err_thres=[(0.25, 2), (0.5, 5), (5, 10)]):
    
    lprint('Testing Pairs: {} Err thres: {}'.format(config.pair_txt, err_thres))
    if sav_res_name is not None:
        save_res_path=os.path.join(config.odir, sav_res_name)
    else:
        save_res_path=None
    print('Evaluate on datasets: {}'.format(test_loaders.keys()))    
    
    net.eval() # Switch to evaluation mode
    abs_err = (-1, -1)
    with torch.no_grad():
        t1 = time.time()
        result_dict = eval_prediction(test_loaders, net, log, pair_type=config.pair_type)
        if sav_res_name is not None:
            np.save(os.path.join(config.odir, 'predictions.npy'),result_dict)
        t2 = time.time()
        print('Total prediction time: {:0.3f}s'.format(t2 - t1))
        abs_err = eval_pipeline_with_ransac(result_dict, log, ransac_thres=config.ransac_thres, 
                                            ransac_iter=10, ransac_miu=1.414, pair_type=config.pair_type, 
                                            err_thres=err_thres, save_res_path=save_res_path)
        print('Total ransac time: {:0.3f}s, Total time: {:0.3f}s'.format(time.time() - t2, time.time() - t1))
    return abs_err

def main():
    # Parse configuration
    config = RelaPoseConfig().parse()
    log = open(config.log, 'a')
    net = networks.__dict__[config.network](config)
    
    # Training/ Testing Datasets 
    if config.training:
        lprint(config2str(config), log)
        datasets = get_datasets(datasets=config.datasets,
                                pair_txt=config.pair_txt,
                                data_root=config.data_root,
                                incl_sces=config.incl_sces, 
                                ops=config.ops,                        
                                train_lbl_txt=config.train_lbl_txt, 
                                test_lbl_txt=config.test_lbl_txt, 
                                with_ess=config.with_ess,
                                with_virtual_pts=config.with_virtual_pts)
        train_set = data.ConcatDataset(datasets)
        lprint('Concat training datasets total samples: {}'.format(len(train_set)), log)
        train_loader = data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                       num_workers=config.num_workers, 
                                       worker_init_fn=make_deterministic(config.seed))
        val_loaders = None
        if config.validate:
            val_loaders = {}
            val_sets = get_datasets(datasets=config.datasets, 
                                    pair_txt=config.val_pair_txt, 
                                    data_root=config.data_root, 
                                    incl_sces=config.incl_sces, 
                                    ops=config.val_ops,
                                    train_lbl_txt=config.train_lbl_txt, 
                                    test_lbl_txt=config.test_lbl_txt, 
                                    with_ess=False)
            for val_set in val_sets:
                val_loaders[val_set.scene] = data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                                                               num_workers=config.num_workers,
                                                               worker_init_fn=make_deterministic(config.seed))
        train(net, config, log, train_loader, val_loaders)
    else:
        lprint('----------------------------------------------\n', log)
        lprint('>>Load weights dict {}'.format(config.weights_dir), log)
        lprint('>>Testing pairs: {}'.format(config.pair_txt), log)

        test_loaders = {}
        test_sets = get_datasets(datasets=config.datasets, 
                                 pair_txt=config.pair_txt, 
                                 data_root=config.data_root, 
                                 incl_sces=config.incl_sces, 
                                 ops=config.ops, 
                                 train_lbl_txt=config.train_lbl_txt, 
                                 test_lbl_txt=config.test_lbl_txt, 
                                 with_ess=False)
        for test_set in test_sets:
            lprint('Testing scene {} samples: {}'.format(test_set.scene, len(test_set)), log)
            test_loaders[test_set.scene] = data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                                                             num_workers=config.num_workers, 
                                                             worker_init_fn=make_deterministic(config.seed))
        test(net, config, log, test_loaders)
    log.close()

if __name__ == '__main__':
    main()
