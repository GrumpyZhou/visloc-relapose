#!/bin/bash

# Training Example
python -m pipeline.relapose_regressor \
        --gpu 0 -b 16 --train -val 20 --epoch 200 \
        --data_root 'data/datasets_480' -ds 'CambridgeLandmarks' \
        --incl_sces 'ShopFacade' \
        -rs 480 --crop 448 --normalize \
        --ess_proj --network 'EssNet' --with_ess\
        --pair 'train_pairs.30nn.medium.txt' -vpair 'val_pairs.5nn.medium.txt' \
        -lr 0.0001 -wd 0.000001 \
        --odir  'output/regression_models/example' \
        -vp 9333 -vh 'localhost' -venv 'main' -vwin 'example.shopfacade'

#  Testing Example
python -m pipeline.relapose_regressor \
        --gpu 2 -b 16  --test \
        --data_root 'data/datasets_480' -ds 'CambridgeLandmarks' \
        --incl_sces 'ShopFacade' \
        -rs 480 --crop 448 --normalize\
        --ess_proj --network 'EssNet'\
        --pair 'test_pairs.5nn.300cm50m.vlad.minmax.txt'\
        --resume 'output/regression_models/example/ckpt/checkpoint_140_0.36m_1.97deg.pth' \
        --odir 'output/regression_models/example'
        
# Test 448 7Scenes
python -m pipeline.relapose_regressor \
        --gpu 2 -b 16  --test \
        --data_root 'data/datasets_480' -ds '7Scenes' \
        -rs 480 --crop 448 --normalize\
        --ess_proj --network 'NCEssNet'\
        --pair 'test_pairs.5nn.5cm10m.vlad.minmax.txt'\
        --resume 'output/regression_models/448_normalize/nc-essnet/7scenes/checkpoint_60_0.04m_1.62deg.pth' \
        --odir 'output/regression_models/448_normalize/nc-essnet/7scenes'


# Test 224 7Scenes
python -m pipeline.relapose_regressor \
    --gpu 2 -b 16  --test \
    --data_root 'data/datasets_256' -ds '7Scenes' \
    -rs 256 --crop 224 \
    --ess_proj --network 'EssNet'\
    --pair 'test_pairs.5nn.5cm10m.vlad.minmax.txt'\
    --resume 'output/regression_models/224_unnormalize/essnet/leverage_datasets/ft_scannet/checkpoint_280_0.02m_0.12deg.pth'\
    --odir 'output/regression_models/224_unnormalize/essnet/leverage_datasets/ft_scannet'
    
# Test relaposenet 224 7Scenes
python -m pipeline.relapose_regressor \
    --gpu 3 -b 16  --test \
    --data_root 'data/datasets_256' -ds '7Scenes' \
    -rs 256 --crop 224 \
    --network 'RelaPoseNet'\
    --pair 'test_pairs.5nn.5cm10m.vlad.minmax.txt'\
    --resume 'output/regression_models/224_unnormalize/relaposenet/7scenes/beta1/4nn_med_lr0.05_lrd0.8-50_wd1e-6/checkpoint_440_0.08m_2.00deg.pth' \
    --odir 'output/regression_models/224_unnormalize/relaposenet/7scenes/beta1/4nn_med_lr0.05_lrd0.8-50_wd1e-6'
    

