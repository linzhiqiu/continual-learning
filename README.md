# The CLEAR Benchmark: Continual LEArning on Real-World Imagery [[www](www.google.com)]

## Introduction
This repository provides code of [*The CLEAR Benchmark*](www.google.com) paper.

<!-- > [**Visual Chirality**](http://bit.ly/visual-chirality),            
> [Zhiqiu Lin](https://linzhiqiu.github.io), [Jin Sun](http://www.cs.cornell.edu/~jinsun/), 
[Abe Davis](http://abedavis.com), [Noah Snavely](https://www.cs.cornell.edu/~snavely/)     
> *IEEE Computer Vision and Pattern Recognition, 2020, Best Paper Nominee*  -->

<!-- For a brief overview of the paper, please check out our oral presentation video!
<p align="center"><a target=_blank href="https://www.youtube.com/watch?v=gc5IvTozU9M&feature=youtu.be"><img src="http://img.youtube.com/vi/gc5IvTozU9M/0.jpg" width="50%" alt="" /></a></p> -->

## Repository Overview

This repository contains all the code and experiments that appear in our paper for reproducibility.

## Structure
<!-- - `train.py`: includes training and validation scripts.
- `config.py`: contains arguments for data preparation, model definition, and imaging details.
- `exp.sh` : contains the experiments script to run.
- All other helper modules :
  - `dataset_factory.py`: prepares PyTorch dataloaders of processed images.
  - `global_setting.py`: contains all supporting demosaicing algorithms and model definitions.
  - `utils.py`: contains functions to generate random images and compute mosiaced/demosaiced/compressed images.
  - `tools.py`: A variety of helpers to get PyTorch optimizer/schedular and logging directory names.

The code is developed using python 3.8.5. NVIDIA GPUs are needed to train and test. -->

# Workflow

## Download YFCC100M Dataset
You can start downloading yfcc100m dataset using scripts provided in this repo. You need to have an AWS account (free to register!) first and enter your access key via:
```
  s3cmd --configure
```
And then you can download yfcc100M metadata to this folder by running:
```
  sh install_yfcc100m.sh
```
To start downloading the YFCC100M images, we provide this python script:
```
  python large_scale_yfcc_download.py --img_dir /data3/zhiqiul/yfcc100m_all_new --min_size 10 --chunk_size 10000 --min_edge 120 --max_aspect_ratio 2;
```
The above script will download the Flickr images with (1) byte size larger than 10, (2) shorter edge larger than 120px, (3) maximum aspect ratio larger than 2 into /data3/zhiqiul. It will split images to multiple subfolders, each containing 10000 (--chunk_size) images. In the meantime, a pickle file will be saved at all_folders.pickle under img_dir. This pickle file after loading is a FlickrFolder() object (see [large_scale_yfcc_download.py](large_scale_yfcc_download.py) for details). All images you downloaded as well as their respective metadata can be accessed by this object.

The downloading speed is not yet optimized, but the all_folders.pickle will be updated after downloading every (--chunk_size) images.

## Segment the temporal stream + CLIP feature extraction

You can start recreate the temporal (uploading) stream of YFCC100M images for already downloaded images from last step, and split these YFCC100M images into a fixed number of buckets with equal size. After splitting the sorted stream into segments, you can generate the CLIP features for each images. These step can by done:
```
  python prepare_dataset.py --img_dir /data3/zhiqiul/yfcc100m_all_new --min_size 10 --chunk_size 10000 --min_edge 120 --max_aspect_ratio 2 --num_of_bucket 11 --model_name RN50x4
```
Note that the above script keeps the same arguments from the downloading step. Additionally, you can specify the number of segments to split by (--num_of_bucket) flag, and the CLIP model used for feature extraction by (--model_name) flag. At this moment, 'RN50', 'RN50x4', 'RN101', and 'ViT-B/32' are available. 'RN50x4' seems to be the best available model. You should check whether OpenAI released new pre-trained models in their [repo](https://github.com/openai/CLIP).


<!-- #### Learning Results with random cropping

With **random cropping**, we can still train network to predict random horizontal reflections on Bayer-demosaiced + JPEG compressed randomly generated gaussian images. We use a cropping size of 512, and in order to eliminate the chance of the network cheating by utilizing the boundary of images (e.g., JPEG edge artifacts), we crop from the center (544, 544) of (576, 576) images. The results again followed our prediction in paper, and they are shown in the following table:

| Image Processing | Image Size | Crop Size| Test Accuracy  |  
|------------------|------------|----|----------------|
| Bayer-Demosaicing| 576 |    512    | 50%  |
| JPEG Compression | 576 |    512    | 50%  | 
| **Both**             | **576** |    **512**   | **99%**  |   -->

<!-- ### Citation
If this work is useful for your research, please cite our paper:
```
@InProceedings{chirality20,
  title={Visual Chirality},
  author = {Zhiqiu Lin and Jin Sun and Abe Davis and Noah Snavely},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
``` -->

### License
This code is freely available for free non-commercial use, and may be redistributed under these conditions. 
Third-party datasets and softwares are subject to their respective licenses. 