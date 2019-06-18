import numpy as np
import torch
from PIL import Image

colors = ['#0F1F90', '#DF6767', '#67DF67','#DFA367', '#6780DF', '#8C6132', '#32798C']
def plot_3d_scatter(vec, label=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vec[:, 0], vec[:, 1], vec[:, 2],s=0.8, c='#125D4C', label=label)
    plt.show()
    
def plot_3d_scatters(data_dict, xlim=(None, None), ylim=(None, None), zlim=(None, None), marker_shape = ">", marker_size = 10, title='', plot_2d=True):    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    #fig.suptitle(title, fontsize=13, fontweight='bold')
    if plot_2d:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')

    fig.subplots_adjust(top=0.98)
    for i, dataset in enumerate(data_dict):
        if i < len(colors):
            color = colors[i]
        else:
            color = np.random.rand(3,)
        vecs = data_dict[dataset]
        if plot_2d:
            ax.scatter(vecs[:, 0], vecs[:, 1], s=marker_size, marker=marker_shape, c=color, label=dataset)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], s=marker_size, marker=marker_shape, c=color, label=dataset)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)                    
    ax.legend()
    plt.show()

def plot_distrb(pos_data, ori_data, range=None, pos_tag='Pos Distance', ori_tag='Ori Distance'):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    axs[0].hist(pos_data, range=range)
    axs[1].hist(ori_data)        
    axs[0].set_title(pos_tag)
    axs[1].set_title(ori_tag)
    
def plot_imlist(imlist):
    '''Plot a list of images in a row'''
    import matplotlib.pyplot as plt
    if type(imlist) is str:
        fig = plt.figure(figsize=(5, 3))
        imlist = [imlist]
    else:
        fig = plt.figure(figsize=(25, 3))
    num = len(imlist)
    for i, im in enumerate(imlist):
        im = Image.open(im)
        ax = fig.add_subplot(1, num, i+1)
        ax.imshow(im)
    plt.show()
    
def torch2rgb(im):
    im = im.squeeze().permute(1, 2, 0)
    if im.device.type == 'cuda':
        im = im.data.cpu().numpy()
    else:
        im = im.data.numpy()
    return im.astype(np.uint8)

def undo_normalize_scale(im):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    im = im * std + mean
    im *= 255.0
    return im.astype(np.uint8)

def plot_pair_loader(data_loader, row_max=2, normalize_and_scale=False):
    import matplotlib.pyplot as plt
    for i, batch in enumerate(data_loader):
        print('>>>>>>>>>')
        fig1 = plt.figure(figsize=(20, 5))
        fig2 = plt.figure(figsize=(20, 5))
        num = len(batch['im_pairs'][0])
        for j in range(num):
            im_pair = batch['im_pairs']
            im1 = im_pair[0][j,:, :, :].permute(1, 2, 0).data.numpy()
            im2 = im_pair[1][j,:, :, :].permute(1, 2, 0).data.numpy()
            if normalize_and_scale:
                im1 = undo_normalize_scale(im1)
                im2 = undo_normalize_scale(im2)
            else:
                im1 = im1.astype(np.uint8)
                im2 = im2.astype(np.uint8)
            ax1 = fig1.add_subplot(1, num, j+1)
            ax1.imshow(im1)
            ax2 = fig2.add_subplot(1, num, j+1)
            ax2.imshow(im2)
        plt.show()
        if i >= row_max:
            break
            
def plot_match_pair_loader(data_loader, normalize_and_scale=False, num_sample=2):
    import matplotlib.pyplot as plt
    
    num = data_loader.batch_size
    count = 0
    for i, batch in enumerate(data_loader):
        print('Batch >>>>>>>>>')
        for j in range(num):
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
            im_src = batch['source_image'][j, :, :, :].permute(1, 2, 0).data.numpy()
            im_pos = batch['pos_target'][j, :, :, :].permute(1, 2, 0).data.numpy()
            im_neg = batch['neg_target'][j, :, :, :].permute(1, 2, 0).data.numpy()
            
            if normalize_and_scale:
                im_src = undo_normalize_scale(im_src)
                im_pos = undo_normalize_scale(im_pos)
                im_neg = undo_normalize_scale(im_neg)
            else:
                im_src = im_src.astype(np.uint8)
                im_pos = im_pos.astype(np.uint8)
                im_neg = im_neg.astype(np.uint8)

                
            axs[0].imshow(im_src)
            axs[1].imshow(im_pos)
            axs[2].imshow(im_neg)
            plt.show()
            count += 1
        if count > num_sample:
            break
            
def plot_matches(src_path, tgt_path, matches, inliers=None, Npts=None, lines=False):
    import matplotlib.pyplot as plt

    # Read images and resize
    I1 = Image.open(src_path)
    I2 = Image.open(tgt_path)
    w1, h1 = I1.size
    w2, h2 = I2.size

    if h1 <= h2:
        scale1 = 1;
        scale2 = h1/h2
        w2 = int(scale2 * w2)
        I2 = I2.resize((w2, h1))
    else:
        scale1 = h2/h1
        scale2 = 1
        w1 = int(scale1 * w1)
        I1 = I1.resize((w1, h2))
    catI = np.concatenate([np.array(I1), np.array(I2)], axis=1)

    # Load all matches
    match_num = matches.shape[0]
    if inliers is None:
        if Npts is not None:
            Npts = Npts if Npts < match_num else match_num
        else:
            Npts = matches.shape[0]
        inliers = range(Npts) # Everthing as an inlier
    else:
        if Npts is not None and Npts < inliers.shape[0]:
            inliers = inliers[:Npts]
        print('Plotting inliers: ', len(inliers))

    x1 = scale1*matches[inliers, 0]
    y1 = scale1*matches[inliers, 1]
    x2 = scale2*matches[inliers, 2] + w1
    y2 = scale2*matches[inliers, 3]
    c = np.random.rand(len(inliers), 3) 

    
    # Plot images and matches
    plt.imshow(catI)
    ax = plt.gca()
    for i, inid in enumerate(inliers):
        # Plot
        ax = plt.gca()
        ax.add_artist(plt.Circle((x1[i], y1[i]), radius=3, color=c[i,:]))
        ax.add_artist(plt.Circle((x2[i], y2[i]), radius=3, color=c[i,:]))
        if lines:
            plt.plot([x1, x2], [y1, y2], c=c[i,:], linestyle='-', linewidth=0.2)
    plt.gcf().set_dpi(350)
    plt.show()