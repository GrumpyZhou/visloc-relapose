#!/bin/bash

# Cambridge 
python -m pipeline.sift_5pt \
        --data_root 'data/datasets_original/' \
        --dataset 'CambridgeLandmarks' \
        --pair_txt 'test_pairs.5nn.300cm50m.vlad.minmax.txt' \
        --cv_ransac_thres 0.5\
        --loc_ransac_thres 5\
        -odir 'output/sift_5pt'\
        -log 'results.dvlad.minmax.txt'

python -m pipeline.sift_5pt \
        --data_root 'data/datasets_original/' \
        --dataset '7Scenes' \
        --pair_txt 'test_pairs.5nn.5cm10m.vlad.minmax.txt' \
        --cv_ransac_thres 0.5 \
        --loc_ransac_thres 15\
        -odir 'output/sift_5pt' \
        -log 'results.dvlad.minmax.txt'
