# Visual Localization Via Relative Camera Pose Estimation

This repository provides implementation of  our paper:   To Learn or Not to Learn: Visual Localization from Essential Matrices [TO UPLOAD]

![Pipeline](pipeline/pipeline.jpg)

To use our code, first download the repository:
````
git clone git@github.com:GrumpyZhou/visloc-relapose.git
````

## Setup Running Environment
We tested the code on Linux Ubuntu 16.04.6 with following versions:
````
1. Python 3.6 + Pytorch 0.4.0  + CUDA 8.0 + CUDNN 8.0v5.1  (Paper version) 
2. Python 3.6 + Pytorch 0.4.0  + CUDA 10.0 + CUDNN 10.0v7.5.1.10
3. Python 3.7 + Pytorch 1.0  + CUDA 8.0 + CUDNN 8.0v5.1
4. Python 3.7 + Pytorch 1.0  + CUDA 10.0 + CUDNN 10.0v7.5.1.10	(Latest version)
````
We recommend to use *Anaconda* to manage packages. Run following lines to automatically setup a ready environment for our code.
````
conda env create -f environment.yml  # This one installs version 3./4.
conda activte relapose
````
Otherwise, one can try to download all required packages seperately according to their offical documentation.

## Prepare Datasets 
Our code is flexible for evaluation on various localization datasets. We use Cambridge Landmarks dataset as an example to show how to prepare a dataset:
1. Create data/ folder
2. Download original [Cambridge Landmarks Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) and extract it to *\$CAMBRIDGE_DIR\$*.
3. Construct folder structure more convenient for running all scripts in this repo:
	````
	cd visloc-relapose/
	mkdir data
	mkdir data/datasets_original
	cd data/original_datasets
	ln -s $CAMBRIDGE_DIR$ CambridgeLandmarks
	````
4. Download [our pairs](https://vision.in.tum.de/webshare/u/zhouq/visloc-datasets/) for training, validation and testing. About the format of our pairs, check [readme](https://vision.in.tum.de/webshare/u/zhouq/visloc-datasets/README.md). 
5. Place pairs to corresponding folder under *data/datasets_original/CambridgeLandmarks*.
6. Pre-save resized 480 images to speed up data loading time (Optional, but Recommended)
	````
	cd visloc-relapose/
	python -m utils.datasets.resize_dataset \
		--base_dir data/datasets_original/CambridgeLandmarks \ 
		--save_dir=data/datasets_480/CambridgeLandmarks \
		--resize 480  --copy_txt True 
	````
7. Test your setup by visualizing the data using [notebooks/data_loading.ipynb](notebooks/data_loading.ipynb) .
8. (Optional) One can also resize the dataset images so that shorter side has 256 pixels at once makes training faster.
#### 7Scenes Datasets
We follow the camera pose label convention of Cambridge Landmarks dataset.  Similarly, you can download  [our pairs](https://vision.in.tum.de/webshare/u/zhouq/visloc-datasets/)  for 7Scenes. For **other datasets**, contact me for information about preprocessing and pair generation.


##  Feature-based: SIFT + 5-Point Solver

We use the SIFT feature extractor and feature matcher in [colmap](https://colmap.github.io/). One can follow the [installation guide](https://colmap.github.io/install.html) to install colmap. We save colmap outputs in database format, see [explanation](https://colmap.github.io/database.html).

### Preparing SIFT features 
Execute following commands to run SIFT extraction and matching on CambridgeLandmarks:
````
cd visloc-relapose/
bash prepare_colmap_data.sh  CambridgeLandmarks
````
Here CambridgeLandmarks is the folder name that is consistent with the dataset folder. So you can also use other dataset names such as 7Scenes if you have prepared the dataset properly in advance.


##  Learning-based: Direct Regression via EssNet


## Hybrid: Learnable Matching + 5-Point Solver
