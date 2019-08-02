#!/bin/bash

# Imagenet + ncn
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset 'CambridgeLandmarks' \
    --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
    --cv_ransac_thres  4.0\
    --loc_ransac_thres 15\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'imagenet+ncn'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/Cambridge'
    
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'imagenet+ncn'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes'

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
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/Cambridge'

python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --feat 'output/regression_models/448_normalize/nc-essnet/7scenes/checkpoint_60_0.04m_1.62deg.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essncn_7sc_60ep+ncn'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes' 

# EssNCNet trained on Cambridge 100ep 
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset 'CambridgeLandmarks' \
    --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
    --cv_ransac_thres 4.0\
    --loc_ransac_thres 15\
    --feat 'output/regression_models/448_normalize/nc-essnet/cambridge/checkpoint_100_0.86m_1.96deg.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essncn_camb_100ep+ncn'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/Cambridge'

python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --feat 'output/regression_models/448_normalize/nc-essnet/cambridge/checkpoint_100_0.86m_1.96deg.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essncn_camb_100ep+ncn'\
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes'

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
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/Cambridge'

python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --feat 'output/pretrained_weights/essnet_scan95ep.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essnet224_scan95+ncn' \
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes'

## EssNet224 on Mega 55ep
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset 'CambridgeLandmarks' \
    --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
    --cv_ransac_thres 4.0\
    --loc_ransac_thres 15\
    --feat 'output/pretrained_weights/essnet_mega55ep.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essnet_mega55ep+ncn' \
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/Cambridge'

python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --feat 'output/pretrained_weights/essnet_mega55ep.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'essnet_mega55ep+ncn' \
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes'


## EssNet224 on MD+7S+CL
python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset 'CambridgeLandmarks' \
    --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
    --cv_ransac_thres 4.0\
    --loc_ransac_thres 15\
    --feat 'output/regression_models/224_unnormalize/essnet/leverage_datasets/ft_mega55/CL_7S/checkpoint_120_0.24m_0.91deg.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'ess224_ftMC7+ncn' \
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/Cambridge'

python -m pipeline.ncmatch_5pt \
    --data_root 'data/datasets_original' \
    --dataset '7Scenes' \
    --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
    --cv_ransac_thres 5.5 \
    --loc_ransac_thres 20 \
    --feat 'output/regression_models/224_unnormalize/essnet/leverage_datasets/ft_mega55/CL_7S/checkpoint_120_0.24m_0.91deg.pth'\
    --ncn 'output/pretrained_weights/nc_ivd_5ep.pth' \
    --posfix 'ess224_ftMC7+ncn' \
    --ncn_thres 0.9 \
    -o 'output/ncmatch_5pt/loc_results/7Scenes'
