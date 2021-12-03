# The CLEAR Benchmark: Continual LEArning on Real-World Imagery [[www](https://clear-benchmark.github.io)]

## Introduction
This repository contains code for preparing [*The CLEAR Benchmark*](https://clear-benchmark.github.io) from [YFCC100M](http://www.yfcc100m.org) and [CLIP](https://github.com/openai/CLIP).

<!-- > [**Visual Chirality**](http://bit.ly/visual-chirality),            
> [Zhiqiu Lin](https://linzhiqiu.github.io), [Jin Sun](http://www.cs.cornell.edu/~jinsun/), 
[Abe Davis](http://abedavis.com), [Noah Snavely](https://www.cs.cornell.edu/~snavely/)     
> *IEEE Computer Vision and Pattern Recognition, 2020, Best Paper Nominee*  -->

<!-- For a brief overview of the paper, please check out our oral presentation video!
<p align="center"><a target=_blank href="https://www.youtube.com/watch?v=gc5IvTozU9M&feature=youtu.be"><img src="http://img.youtube.com/vi/gc5IvTozU9M/0.jpg" width="50%" alt="" /></a></p> -->

## Repository Overview

This repository contains all the code snippets for CLEAR dataset curation as detailed in our paper for reproducibility. Specifically you will find code for:

- YFCC100M metadata and image download
- CLIP feature extraction from downloaded images
- Visio-linguistic dataset curation via CLIP-based image retrieval
- MoCo V2 model training on downloaded images

<!-- ## Structure -->
<!-- - `train.py`: includes training and validation scripts.
- `config.py`: contains arguments for data preparation, model definition, and imaging details.
- `exp.sh` : contains the experiments script to run.
- All other helper modules :
  - `dataset_factory.py`: prepares PyTorch dataloaders of processed images.
  - `global_setting.py`: contains all supporting demosaicing algorithms and model definitions.
  - `utils.py`: contains functions to generate random images and compute mosiaced/demosaiced/compressed images.
  - `tools.py`: A variety of helpers to get PyTorch optimizer/schedular and logging directory names.

The code is developed using python 3.8.5. NVIDIA GPUs are needed to train and test. -->

# YFCC Download and Feature Extraction

## Download YFCC100M Dataset
You can start downloading YFCC100M dataset using scripts provided in this repo. You need to have an AWS account ([free to register!](](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwi8g--W27L0AhVPdGAKHTO7AzkYABABGgJ0bQ&ae=2&ohost=www.google.com&cid=CAESQOD2YorSMEakEBdFNCteMjdgGcrq_dZOeB9OQ8ex8Wb3cUKCQ6H04wAuXLoPwD1dryftDz3eMgrQGM1ck4HBhp8&sig=AOD64_2p2IufhX7iauC2Oh-Y3K56sKdM9g&q&adurl&ved=2ahUKEwjIwOaW27L0AhVTIqYKHUIYDywQ0Qx6BAgEEAE&dct=1))) first and enter your access key via:
```
  s3cmd --configure
```
And then you can download YFCC100M metadata (e.g., download url, upload timestamp, user tags, ..) to current folder by running:
```
  sh install_yfcc100m.sh
```
<!-- To start downloading the YFCC100M images, we provide this python script (for single-threaded download): -->
<!-- ```
  python large_scale_yfcc_download.py --img_dir /data3/zhiqiul/yfcc100m_all_new --min_size 10 --chunk_size 10000 --min_edge 120 --max_aspect_ratio 2;
``` -->
Assume you download the metadata in current folder, then to download YFCC100M images, simply run this python script (for multi-threading download with parallel workers):
```
  python yfcc_download.py
```

You should provide the below arguments:
- *--img_dir* : 
  - Path to save the images, e.g., /path/to/save/
- *--metadata_dir* (default = ./): 
  - Path with downloaded metadata files from last step 
- *--min_size* (default = 10): 
  - If set to 0, download images with arbitrary size. Otherwise, only download images larger than **min_size** byte size
  - Recommended to set to 10 because some invalid images are smaller than 10 bytes.
- *--min_edge* (default = 0) : 
  - If set to 0, then download images with arbitrary height/width. Otherwise, only download images with edges larger than **min_edge** pixels
- *--max_aspect_ratio* (default = 0): 
  - If set to 0, then download images with any aspect ratios; otherwise, only download images with aspect ratios smaller than **max_aspect_ratio**
- *--chunk_size* (default = 50000): 
   - In order not to save too many files under a single folder, the images will be splitted and saved at subfolders under **img_dir**, i.e., 0-th to **(chunk_size - 1)**-th images will be downloaded under **img_dir/0**, **(chunk_size)**-th to **(2 x chunk_size - 1)**-th images will be downloaded under **img_dir/1**, ... 
- *--use_valid_date* (default = True):
   - If set to True, the do not download images with captured timestamp later than upload timestamp.
   - Recommended to set to True for sanity.
- *--max_workers* (default = 128):
   - The number of parallel workers for multi-threading download.

An example script is:
```
  python yfcc_download.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --metadata_dir ./ --min_size 10 --min_edge 120 --max_aspect_ratio 2 --chunk_size 50000
```



The above script will download the Flickr images with (1) byte size larger than 10 (**min_size**), (2) shorter edge larger than 120 pixels (**min_edge**), (3) maximum aspect ratio larger than 2 (**max_aspect_ratio**). It will split images to multiple subfolders indexed by numbers under **img_dir**, each containing at most 50000 (**chunk_size**) images. You can stop the script anytime once you have downloaded enough images.

If you run this script, a pickle file will be saved and updated at **img_dir/all_folders.json**. All images you downloaded as well as their respective metadata can be accessed by this object. **Do not delete this file at anytime since it keeps track of the download status.**

Caveat: If your RAM is limited, the script might be killed occasionally. In that case, you just need to rerun the same script and it will resume from the previous checkpoint.

<!-- To download at full speed (which requires more RAM resources), we also provide a multi-threading version of the same python script:
```
  python large_scale_yfcc_download_parallel.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2;
``` -->
<!-- The above script will download the Flickr images with parallel workers (MAX_WORKERS=128 as default). It will also split images to multiple subfolders, each containing at most 50000 (--chunk_size) images. The rest are the same as the above script. Note that if your RAM is limited, the script might be killed occasionally. In that case, you just need to rerun the script and it will resume from the previous checkpoint. -->
## Segment the temporal stream + CLIP feature extraction

You can recreate the temporal (uploading) stream of YFCC100M images for already downloaded images by [prepare_dataset.py](prepare_dataset.py). To use this script, you should supply with the same list of arguments from last step, plus three additional arguments:
- *--split_by_year* (default = False):
   - If set to True, split the image (upload) stream to 11 buckets by year from 2004 to 2014 (note that each year may have different number of images). If set to False, then evenly split to **num_of_buckets** buckets (see next argument). 
- *--split_by_time* (default = None):
   - By default it is None. Or you can set it to a json file with the desired time period for each bucket. You can find an example for such json file at [clear_10_time.json](clear_10_time.json).
- *--num_of_bucket* (default = 11):
   - If **split_by_year** is set to False and **split_by_time** is set to None, then split to **num_of_bucket** equal-sized buckets.
- *--model_name* (default = RN50):
   - The name of pre-trained CLIP model.
   - For now, we only support 'RN50', 'RN50x4', 'RN101', and 'ViT-B/32'. You may check whether OpenAI have released new pre-trained models in their [repo](https://github.com/openai/CLIP).

Here is an example script following the arguments from last step to split images by year:
```
  python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year True --model_name RN101
```
The above script will (1) sort the downloaded YFCC100M images by upload timestamp, (2) Split the stream by year from 2004 to 2014, (3) generate the CLIP features for each image in each bucket with RN101 (resnet101) model. After the script is finished, you can find a json file containing all the bucket information at **img_dir**/bucket_by_year.json, for example:
```
/scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_by_year.json
```

You can also split to equal-sized buckets via:
```
  python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year False --num_of_buckets 11 --model_name RN101
```
This will produce a json file at **img_dir**/bucket_**num_of_buckets**.json, for example:
```
/scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_11.json
```
Additionally, you can also specify the time period for each segment in a json file such as [clear_10_time.json](clear_10_time.json):
```
  python prepare_dataset.py --img_dir /scratch/zhiqiu/yfcc100m_all_new_sep_21 --min_size 10 --chunk_size 50000 --min_edge 120 --max_aspect_ratio 2 --split_by_year False --split_by_time ./clear_10_time.json --model_name RN101
```
This will produce a json file at **img_dir**/bucket_**name_of_json_file**.json, for example:
```
/scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_clear_10_time.json
```

<!-- If you need to move the downloaded files to another location before you start running any experiments, you can specify the (--new_folder_path) flag. The reason that you must use this script to transfer the folder is because you cannot simply copy the downloaded folders: The metadata objects contain absolute paths to the image files. Check out the comments in [prepare_dataset.py](prepare_dataset.py) for more details. -->

# Visio-linguistic dataset curation with CLIP
## Prompt Engineering with CLIP
Once you download the dataset and extract the CLIP features, you can use the interactive jupyter notebook ([CLIP-PromptEngineering.ipynb](CLIP-PromptEngineering.ipynb)) to perform image retrival and try out different prompts for your visual concepts of interest! Please follow the instruction in the notebook and change to your local path to bucket_dict json file generated from last step. 

## Image Retrieval with a group of visual concepts
Once you find a list of engineered prompts for different visual concepts, you may retrieve images for all the prompts at once with [prepare_concepts.py](prepare_concepts.py). Before doing so, you should input the prompts to a json file as well as specifying all parameters for collecting the dataset. One such example is [clear_10_config.json](clear_10_config.json), and here is an explanation for all the configurations:
```
{
    "NAME" : "CLEAR10-TEST", # Name your own dataset
    "PREFIX" : "", # A prefix to all visual concepts, such as "a photo of"
    "ALLOW_OVERLAP" : 0, # If set to 0, images appear in multiple categories will be removed. Otherwise, keep all top-scoring images retrieved per concept.
    "BUCKET_DICT_PATH" : "/scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_11.json", # The path to the bucket_dict created by prepare_dataset.py
    "CLIP_MODEL" : "RN101", # The pre-trained model used when extracting CLIP features
    "NUM_OF_IMAGES_PER_CLASS_PER_BUCKET" : 600, # The number of images to retrieve per class per bucket
    "NUM_OF_IMAGES_PER_CLASS_PER_BUCKET_TO_QUERY" : 16000, # The number of images to query (this number must be larger than the above).
    "BACKGROUND" : 1, # If set to 1, add an additional background class consisting of negative samples collected per concept. If set to 0, then no background class will be collected.
    "NEGATIVE_RATIO" : 0.1, # The ratio of negative samples per class to keep
    "SAVE_PATH" : "/data3/zhiqiul/clear_datasets", # The images will be saved at this folder.
    "GROUP" : # A group of prompts for each visual concept
    [
        "laptop",
        "camera",
        "bus",
        "sweater",
        "dress",
        "racing",
        "hockey",
        "cosplay",
        "baseball",
        "soccer"
    ]
}
```
After you modify this json file (or create your own), you can start collecting via:
```
  python prepare_concepts.py --concept_group_dict ./clear_10_config.json
```
The retrieved images will then be saved under **SAVE_PATH/NAME/**.

## CSV files preparation
You can export the metadata to CSV files via prepare_csv.py.

## MoCo V2 Pre-training
You can pre-train a MoCo V2 model via scripts under [moco/](moco/) folder. After running the above steps, you can specify a bucket of the temporal YFCC100 image stream to pre-train a MoCo V2 model. For more details about training MoCo, please refer to their [official repository](https://github.com/facebookresearch/moco). For example, we can use the default MoCo V2 hyperparameter to pre-train a MoCo model using the 0th bucket from the previous step (you need to modify the --data flag to your local file location that saves the bucket of image metadata; and modify the --model_folder to where you want the MoCo V2 model to be saved):
<!-- ```
  python moco/main_yfcc.py --data /scratch/zhiqiu/yfcc100m_all_new/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_11/0/bucket_0.pickle --model_folder /data3/zhiqiul/yfcc_moco_models/july_6_bucket_11_idx_0_gpu_8/ --arch resnet50 -j 32 --lr 0.03 --batch-size 256 --dist-url 'tcp://localhost:10023' --multiprocessing-distributed --mlp --moco-t 0.2 --aug-plus --cos
``` -->
```
  python moco/main_yfcc.py --data /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_by_year/2004/bucket_0.json --model_folder /data3/zhiqiul/yfcc_moco_models/sep_21_year_2004_gpu_8/ --arch resnet50 -j 32 --lr 0.03 --batch-size 256 --dist-url 'tcp://localhost:10023' --multiprocessing-distributed --mlp --moco-t 0.2 --aug-plus --cos
```
The above script requires 8 (RTX 2080) GPUs. You can shrink the batch size if you have fewer available GPUs.
## Data Cleaning with MTurk/Manual Inspection
TODO.

## Classifier Training.
TODO. Maybe work with avalanche.
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