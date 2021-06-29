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