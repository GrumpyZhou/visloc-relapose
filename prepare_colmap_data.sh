#!/bin/bash

############################### USAGAGE ################################
# This script preforms SIFT feature extraction and feature mathing     #
# for the specified dataset. Colmap outputs are saved in corresponding #
# database.db files under database root dir: data/colmap_dbs.          #
# Available dataset options:  CambridgeLandmarks, 7Scenes.             #
# Example:                                                             #
#    bash  prepare_colmap_data.sh  CambridgeLandmarks                  # 
########################################################################

DATASET=$1
DATASET_ROOT=data/datasets_original
COLMAP_DB_ROOT=data/colmap_dbs
echo 'Colmap preprocessing....'
echo 'COLMAP_DB_ROOT:' $COLMAP_DB_ROOT 
echo 'DATASET_ROOT:' $DATASET_ROOT 
echo 'DATASET:' $DATASET

if [ "$DATASET" == "CambridgeLandmarks" ]; then
    SCENES=("ShopFacade" "OldHospital" "KingsCollege" "StMarysChurch")
else
    SCENES=("heads" "chess" "fire" "office" "pumpkin" "redkitchen" "stairs")
fi

for SC in "${SCENES[@]}"
do
    # Prepare images.txt and database.db for colmap outputs
    python -m utils.colmap.write_images  \
           --data_root $DATASET_ROOT --colmap_db_root $COLMAP_DB_ROOT\
           --dataset $DATASET --scene $SC

    IMG_PATH=$DATASET_ROOT/$DATASET/$SC
    DATABASE_PATH=$COLMAP_DB_ROOT/$DATASET/$SC
    echo 'IMG PATH: ' $IMG_PATH 
    echo 'DATABASE PATH' $DATABASE_PATH

    # Feature Extraction
    colmap feature_extractor \
        --database_path $DATABASE_PATH/database.db \
        --ImageReader.camera_model PINHOLE \
        --image_list_path $DATABASE_PATH/images.txt \
        --image_path $IMG_PATH \

    # Feature Matching
    colmap exhaustive_matcher \
       --database_path $DATABASE_PATH/database.db 
done