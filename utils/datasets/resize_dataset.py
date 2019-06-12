import argparse
import os
import numpy as np
from PIL import Image
from shutil import copyfile
import glob

def resize_datasets(base_dir, scenes, resize, save_base_dir, copy_txt=True):
    """
    Resize given dataset images and save to a new dir.
    The dataset needs to contain dataset_train.txt/dataset_test.txt.
    The new dataset follows the same structure as the original one.
    Setting copy_txt to true will copy *.txt to the new dir.
    """
    
    train_txt='dataset_train.txt'
    test_txt='dataset_test.txt'
    for scene in scenes:
        data_dir = os.path.join(base_dir, scene)
        print('Resize dataset {} to size {}'.format(data_dir, resize))        
        save_dir = os.path.join(save_base_dir, scene)
        dt_txt = os.path.join(data_dir, train_txt)
        num = 0 
        with open(dt_txt, 'r') as f:
            # im x y z w p q r
            for i,line in enumerate(f):
                if i < 3:
                    continue
                im_name = line.split()[0]
                im = Image.open(os.path.join(data_dir, im_name))
                if np.array(im).shape[-1] > 3:  # If it is a RBGA image, keep only RGB
                    im = Image.fromarray(np.array(im)[:, :, 0:3])
                pdir = os.path.dirname(os.path.join(save_dir, im_name))
                if not os.path.exists(pdir):
                    os.makedirs(pdir)
                    print('mkdir {}'.format(pdir))
                if resize is not None:
                    im = resize_(im, resize, interpolation=Image.BICUBIC)
                im.save(os.path.join(save_dir, im_name))
                num += 1
            print('Processed train images {}'.format(num))

        dt_txt = os.path.join(data_dir, test_txt)
        num = 0 
        with open(dt_txt, 'r') as f:
            # im x y z w p q r
            for i,line in enumerate(f):
                if i < 3:
                    continue
                im_name = line.split()[0]
                im = Image.open(os.path.join(data_dir, im_name))
                if np.array(im).shape[-1] > 3:  # If not RGB keep only the first 3 channels
                    im = Image.fromarray(np.array(im)[:, :, 0:3])
                pdir = os.path.dirname(os.path.join(save_dir, im_name))
                if not os.path.exists(pdir):
                    os.makedirs(pdir)
                    print('mkdir {}'.format(pdir))
                if resize is not None:
                    im = resize_(im, resize, interpolation=Image.BICUBIC)
                im.save(os.path.join(save_dir, im_name))
                num += 1
            print('Processed test images {}'.format(num))
        
        if copy_txt:
            copied = []
            for path in glob.iglob(os.path.join(data_dir, '*.txt')):
                file_name = os.path.basename(path)
                copied.append(file_name)
                copyfile(os.path.join(data_dir, file_name), os.path.join(save_dir, file_name))
            print('Copied files: {}'.format(copied))
            
def resize_(im, size, interpolation=Image.BICUBIC):
    """
    Resize an image so that the shorter side 
    is the same as the given size and the 
    aspect ratio remains the same.
    """
    w, h = im.size
    if (w <= h and w == size) or (h <= w and h == size):
        return im
    if w < h:
        ow = size
        oh = int(size * h / w)
        return im.resize((ow, oh), interpolation)
    else:
        oh = size
        ow = int(size * w / h)
        return im.resize((ow, oh), interpolation)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',  type=str)
    parser.add_argument('--save_dir', type=str)                  
    parser.add_argument('--scenes', '-sc', type=str, nargs='*', default=None)
    parser.add_argument('--resize', type=int, default='480')
    parser.add_argument('--copy_txt', type=bool, default=True)
    args = parser.parse_args()
    
    if args.scenes is None:
        if 'CambridgeLandmarks' in args.base_dir:
            args.scenes = ['KingsCollege', 'OldHospital', 'ShopFacade',  'StMarysChurch'],
        if '7Scenes' in args.base_dir:
            args.scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    print(args)
    resize_datasets(args.base_dir, args.scenes, args.resize, args.save_dir, args.copy_txt)

if __name__ == '__main__':
    main()