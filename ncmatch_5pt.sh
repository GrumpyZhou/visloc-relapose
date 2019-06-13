#!/bin/bash

#### Test hybrid
# Imagenet + ncn
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset 'CambridgeLandmarks' \
    --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
    --cv_ransac_thres  4.0\
    --loc_ransac_thres 15\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --match_save_root 'output/ncmatch_5pt/saved_matches'\
    --posfix 'imagenet+ncn'\
    --ncn_thres 0.9 \
    --gpu 1\
    -o 'output/ncmatch_5pt/loc_results/Cambridge/imagenet+ncn.txt'
    
# 103268
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --match_save_root 'output/ncmatch_5pt/saved_matches'\
    --posfix 'imagenet+ncn'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes/imagenet+ncn.txt'


# EssNCNet trained on 7Scenes 60ep
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset 'CambridgeLandmarks' \
    --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
    --cv_ransac_thres 4.0\
    --loc_ransac_thres 15\
    --feat 'output/regression_models/448_normalize/nc-essnet/7scenes/checkpoint_60_0.04m_1.62deg.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \    
    --posfix 'essncn_7sc_60ep+ncn'\
    --match_save_root 'output/ncmatch_5pt/saved_matches'\
    --ncn_thres 0.9 \
    --gpu 2\
    -o 'output/ncmatch_5pt/loc_results/Cambridge/essncn_7sc_60ep+ncn.txt' 

# 103269
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --feat 'output/regression_models/448_normalize/nc-essnet/7scenes/checkpoint_60_0.04m_1.62deg.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essncn_7sc_60ep+ncn'\
    --match_save_root 'output/ncmatch_5pt/saved_matches'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes/essncn_7sc_60ep+ncn.txt' 

## EssNet224 on Scannet 95ep
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset 'CambridgeLandmarks' \
    --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
    --cv_ransac_thres 4.0\
    --loc_ransac_thres 15\
    --feat 'output/pretrained_weights/essnet_scan95ep.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essnet224_scan95+ncn' \
    --match_save_root 'output/ncmatch_5pt/saved_matches'\
    --ncn_thres 0.9 \
    --gpu 3 \
    -o 'output/ncmatch_5pt/loc_results/Cambridge/ess224_scan95+ncn.txt' 

# 103270
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --feat 'output/pretrained_weights/essnet_scan95ep.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essnet224_scan95+ncn' \
    --match_save_root 'output/ncmatch_5pt/saved_matches'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes/7Scenes/ess224_scan95+ncn.txt'


#### Train
# >>>>>>>>>> >>>>>>>>>> >>>>>>>>>> IVD
# ImmMatch  Scratch
# python -m run.immatch.train	\
#     -b 64 --train -val 1 --epoch 10 \
#     --data_root 'data/datasets_original/ivd' \
#     -rs 256 -c 224 -norm  --hflip 0.5 \
#     -tdict 'data/origin_datasets/ivd/image_pairs/ivd_train_pairs.neg10.npy' \
#     -vdict 'data/origin_datasets/ivd/image_pairs/ivd_val_pairs.neg10.npy' \
#     --optim 'Adam' -lr 0.0005 -wd 0.0 \
#     --save_inter_ep  0\
#     --odir 'output/immatch/ivd/bz64_224crop_lr5e-4'


