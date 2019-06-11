import os
import argparse
from utils.colmap.init_database import COLMAPDatabase

def prepare_scene_imlist(dataset, scene, dataset_root, colmap_db_root):
    """Pre-steps for colmap sift feature extraction:
        1. Create corresponding colmap db folder
        2. Create an empty colmap db for sift output
        3. Collect dataset images from dataset_train.txt and dataset_test.txt
        4. Write into images.txt and save to corresponding colmap db folder
    """
    
    dataset_dir = os.path.join(dataset_root, dataset)
    colmap_db_dir = os.path.join(colmap_db_root, dataset)
    print('>>>Prepare Dataset: {} Scene: {}'.format(dataset_dir, scene))
    src_dir = os.path.join(dataset_dir, scene)
    dst_dir = os.path.join(colmap_db_dir, scene)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Create database.db
    database_path = os.path.join(dst_dir, 'database.db')
    if not os.path.exists(database_path):
        print('Initialize colmap db: {}'.format(database_path))
        db = COLMAPDatabase.connect(database_path)
        db.close()

    # Merge train and test set into images.txt
    train_txt = os.path.join(src_dir, 'dataset_train.txt')
    test_txt = os.path.join(src_dir, 'dataset_test.txt')
    images_txt = os.path.join(dst_dir, 'images.txt')
    train = open(train_txt, 'r')
    test = open(test_txt, 'r')
    images = open(images_txt, 'w')
    print('Write images to {}'.format(images_txt))

    # Copy training images
    count = 0
    for line in train.readlines()[3::]:
        imname = line.split()[0]
        images.write('{}\n'.format(imname))
        count += 1
    print('Training images ', count)

    count = 0
    for line in test.readlines()[3::]:
        imname = line.split()[0]
        images.write('{}\n'.format(imname))
        count += 1
    print('Testing images ', count)
    train.close()
    test.close()
    images.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/datasets_original')
    parser.add_argument('--colmap_db_root', type=str, default='data/colmap_dbs')
    parser.add_argument('--dataset', choices=['CambridgeLandmarks', '7Scenes'], type=str)
    parser.add_argument('--scene',  type=str)
    args = parser.parse_args()
    prepare_scene_imlist(args.dataset, args.scene, args.data_root, args.colmap_db_root)