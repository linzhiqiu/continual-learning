# The CLEAR Benchmark: Continual LEArning on Real-World Imagery [[www](https://clear-benchmark.github.io)]

## Introduction
This repository contains code for preparing [*The CLEAR Benchmark*](https://clear-benchmark.github.io) from [YFCC100M](http://www.yfcc100m.org) and [CLIP](https://github.com/openai/CLIP).

## Repository Overview

This repository contains all the code snippets for CLEAR dataset curation as detailed in our paper for reproducibility. Specifically you will find code for:

- YFCC100M metadata and image download
- CLIP feature extraction from downloaded images
- Visio-linguistic dataset curation via CLIP-based image retrieval
- MoCo V2 model training on downloaded images

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

If you run this script, a json file will be saved and updated at **img_dir/all_folders.json**. All images you downloaded as well as their respective metadata can be accessed through this dictionary. **Do not delete this file at anytime since it keeps track of the current download status.**

Caveat: If your RAM is limited, the script might be killed occasionally. In that case, you just need to rerun the same script and it will resume from the previous checkpoint.

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

You can also split image stream to equal-sized buckets (like what we did in paper) via:
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

# Visio-linguistic dataset curation with CLIP
## Prompt Engineering with CLIP
Once you download the dataset and extract the CLIP features following the instruction above, you can use the interactive jupyter notebook ([CLIP-PromptEngineering.ipynb](CLIP-PromptEngineering.ipynb)) to perform image retrival and try out different prompts for your visual concepts of interest! Please follow the instruction in the notebook and change to your local path to the json file generated from last step. 

## Image Retrieval with a group of visual concepts
After you find a list of engineered prompts for different visual concepts, you may retrieve images for all the prompts at once with [prepare_concepts.py](prepare_concepts.py). Before doing so, you should input the prompts to a json file as well as specifying all parameters for collecting the dataset. One such example is [clear_10_config.json](clear_10_config.json), and here is an explanation for all the configurations:
```
{
    # Name your own dataset
    "NAME" : "CLEAR10-TEST",

    # A prefix to all visual concepts, such as "a photo of"
    "PREFIX" : "", 
    
    # If set to 0, images appear in multiple categories will be removed. Otherwise, keep all top-scoring images retrieved per concept.
    "ALLOW_OVERLAP" : 0, 

    # The path to the bucket_dict created by prepare_dataset.py
    "BUCKET_DICT_PATH" : "/scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_11.json",

    # The pre-trained model used when extracting CLIP features
    "CLIP_MODEL" : "RN101", 

    # The number of images to retrieve per class per bucket
    "NUM_OF_IMAGES_PER_CLASS_PER_BUCKET" : 600,

    # The number of images to query (this number must be larger than the above)
    "NUM_OF_IMAGES_PER_CLASS_PER_BUCKET_TO_QUERY" : 16000,

    # If set to 1, add an additional background class consisting of negative samples collected per concept. If set to 0, then no background class will be collected.
    "BACKGROUND" : 1, 

    # The ratio of negative samples retrieved per class to keep
    "NEGATIVE_RATIO" : 0.1,

    # The images will be saved at this folder.
    "SAVE_PATH" : "/path/to/data/",

    # A group of prompts for each visual concept
    "GROUP" :
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
After you modify this [json file](clear_10_config.json) (or create your own), you can retrieve the images via:
```
  python prepare_concepts.py --concept_group_dict ./clear_10_config.json
```
Optionally, if the prompts you designed are too long, you can provide short names for them in a dictionary (key is prompt, value is the short name string) in json file via the optional **--label_map_dict** parameter.

The retrieved images will then be saved under **SAVE_PATH/NAME/labeled_images**. Metadata for labeled images will be saved under **SAVE_PATH/NAME/labeled_metadata/**, you may access the file names via bucket index and label name via **SAVE_PATH/NAME/labeled_metadata.json**. When determining the class index, class names will be sorted via alphabetical order, stored per line in **SAVE_PATH/NAME/class_names.txt**.

Finally, if you want to save the raw images/metadata for all images, you can set **--save_all_images** or **--save_all_metadata** to be **True**. Then raw images/metadata for all images per bucket are saved under **SAVE_PATH/NAME/all_images/** and **SAVE_PATH/NAME/all_metadata/**, and metadata file names per bucket is saved in **SAVE_PATH/NAME/all_metadata.json**. For storage concerns, we recommand you to set **--save_all_metadata** to **True** and **--save_all_images** to **False**; if you want to download all images, you can always run [download_all_images.py](download_all_images.py) later.

The resulting folder looks like:

```
SAVE_PATH/NAME/
|   concept_group_dict.json (a copy for reference)
|   class_names.txt (line number for each class is the class index)
|   clip_result.json (clip retrieved scores and metadata for all buckets)
|   labeled_metadata.json (mapping bucket index to labeled metadata path)
|   all_metadata.json (mapping bucket index to all metadata path, optional)
└───labeled_images (for clip-retrieved images)
|   └───0
|   |   └───computer
|   |   |   |   235821044.jpg
|   |   |   |   ...
|   |   └───camera
|   |   |   |   269400202.jpg
|   |   |   |   ...
|   |   └───...
|   └───1
|   └───...
└───labeled_metadata (metadata for labeled images)
|   └───0
|   |   |   computer.json (dict: key is flickr ID, value is metadata dict)
|   |   |   camera.json
|   |   |   ...
|   └───1
|   |   |   ...
|   └───...
└───all_images (all images in a bucket, optional)
|   └───0
|   |   |   648321013.jpg
|   |   |   ...
|   └───1
|   └───...
└───all_metadata (metadata for all images, optional)
│   │   0.json
|   |   1.json
|   |   ...
```

## MoCo V2 pre-training on single bucket
You can pre-train a MoCo V2 model via scripts under [moco/](moco/) folder. Specifically, after you download the images and segment them into buckets, you can specify a bucket from the stream to pre-train a MoCo V2 model. For more details about training MoCo and tuning hyperparameters, please refer to their [official repository](https://github.com/facebookresearch/moco). As an example, we can use the default MoCo V2 hyperparameter to pre-train a MoCo model using the 0th bucket from the previous step (you need to modify the **--data** flag to your local file location that saves the bucket of image metadata; and modify the **--model_folder** to where you want the MoCo V2 model to be saved):
```
  python moco/main_yfcc.py --data /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_11/0/bucket_0.json --model_folder /path/to/model --arch resnet50 -j 32 --lr 0.03 --batch-size 256 --dist-url 'tcp://localhost:10023' --multiprocessing-distributed --mlp --moco-t 0.2 --aug-plus --cos
```
<!-- ```
  python moco/main_yfcc.py --data /scratch/zhiqiu/yfcc100m_all_new_sep_21/images_minbyte_10_valid_uploaded_date_minedge_120_maxratio_2.0/bucket_11/0/bucket_0.json --model_folder /data3/zhiqiul/yfcc_moco_models/sep_21_bucket_0_gpu_8/ --arch resnet50 -j 32 --lr 0.03 --batch-size 256 --dist-url 'tcp://localhost:10023' --multiprocessing-distributed --mlp --moco-t 0.2 --aug-plus --cos
``` -->
The above script requires 8 (RTX 2080) GPUs. You can shrink the batch size if you have fewer available GPUs. Checkpoint will be saved at the end of every epoch of training; e.g., if you want to load the checkpoint at the end of epoch 200th, the path would be **model_folder**/checkpoint_0199.pth.tar.

If you want to use the model checkpoint for feature extract (see next section), you may run the below script to extract the 'state_dict' in order to load into a pytorch initialized ResNet model. For example:
```
  python adapt_moco_model.py --model_path /path/to/model/checkpoint_0199.pth.tar --save_path /path/to/model/best_state_dict.pth.tar
```
<!-- ```
  python adapt_moco_model.py --model_path /data3/zhiqiul/yfcc_moco_models/sep_21_bucket_0_gpu_8/checkpoint_0199.pth.tar --save_path /data3/zhiqiul/yfcc_moco_models/sep_21_bucket_0_gpu_8/best_state_dict.pth.tar
``` -->


## Extract features for training
Once you pre-train your MoCo model (or any other unsupervised model and save its state_dict), you may extract features via [prepare_features.py](prepare_features.py):
```
  python prepare_features.py --folder_path /path/to/data/ --name moco_b0 --state_dict_path /path/to/model/best_state_dict.pth.tar --arch resnet50
```
<!-- ```
  python prepare_features.py --folder_path /data3/zhiqiul/clear_datasets/CLEAR10-TEST --name moco_b0 --state_dict_path /data3/zhiqiul/yfcc_moco_models/sep_21_bucket_0_gpu_8/best_state_dict.pth.tar --arch resnet50
``` -->
You should change the arguments based on your local paths to the model checkpoint and image folders. These arguments include:
- *--folder_path*:
   - The folder generated by prepare_concepts.py (containing labeled_metadata.json, etc.). Features will also be saved under this folder. 
- *--name* (default = 'default'):
   - You can name your model via this argument. Features will be saved under **folder_path**/features/**name**.
- *--state_dict_path*:
   - You should specify the path to model state_dict here.
- *--arch* (default='resnet50'):
   - You should specify the architecture of this model checkpoint here.

Note that by default, the images will undergo a fixed imagenet transformation scheme: (1) Resize(224), (2) CenterCrop(224), (3) ToTensor(), (4) Normalize ( imagenet mean and std). Feel free to explore other transformations and rename the **model_name** accordingly.

The resulting directory tree looks like:
```
SAVE_PATH/NAME/
|   ... (other files/folders generated by prepare_concepts.py)
|   class_names.txt (line number for each class is the class index)
|   labeled_metadata.json (mapping bucket index to labeled metadata path)
└───labeled_metadata (metadata for labeled images)
|   └───0
|   |   |   computer.json (dict: key is flickr ID, value is metadata dict)
|   |   |   camera.json
|   |   |   ...
|   └───1
|   |   |   ...
|   └───...
└───features (features for labeled images)
|   └───{args.name}
|   |   |   features.json (mapping bucket index to features pth files, similar to labeled_metadata.json)
|   |   |   state_dict.pth.tar (a copy of the state_dict file)
|   |   └───0
|   |   |   |   computer.pth (dict: key is flickr ID, value is torch tensor)
|   |   |   |   camera.pth
|   |   |   |   ...
|   |   └───1
|   |   |   |   ...
|   |   └───...
```

# Prepare folders for Avalanche training
Once we finish the above steps, you can run [prepare_training_folder.py](prepare_training_folder.py) to prepare the folder ready for Avalanche training. In particular, [Avalanche](https://avalanche.continualai.org) (an well-maintained end-to-end library for CL training and evaluation) supports reading in (1) caffe-style filelists or (2) pytorch tensor lists when creating benchmark objects. Therefore, [prepare_training_folder.py](prepare_training_folder.py) supports two functionalities:

- Store filelists and tensor lists in format supported by avalanche.
- Generate random train/test split based on given a random seed number and a split ratio.

Suppose the folder from last step is stored at **SAVE_PATH/NAME**, you can save all tensor lists and filelists without train/test split via:
```
  python prepare_training_folder.py --folder_path SAVE_PATH/NAME
```

Or you can specify the **testset_ratio** parameter to perform train/test split using seeds in `SEED_LIST` at the top of [prepare_training_folder.py](prepare_training_folder.py) (currently it runs for 0,1,2,3,4 a total of five seeds, but feel free to edit for other seeds):
```
  python prepare_training_folder.py --folder_path SAVE_PATH/NAME --testset_ratio 0.3
```

After running any of the above scripts, a new folder named **training_folder** will be saved under **SAVE_PATH/NAME** as below:

```
SAVE_PATH/NAME
|   ... (other files/folders generated by prepare_concepts.py)
└───features (suppose we have two feature types imagenet and moco_b0)
|   └───imagenet (same structure as above)
|   └───moco_b0 (same structure as above)
└───training_folder
|   bucket_indices.json (a list of bucket indices)
|   └───features (saving all (feature tensors, class index) in avalanche tensor list format)
|   |   └───imagenet
|   |   |   └───0
|   |   |   |   |   all.pth (feature tensors, labels)
|   |   |   └───1
|   |   |   └───...
|   |   └───moco_b0
|   |   |   └───0
|   |   |   |   |   all.pth (feature tensors, labels)
|   |   |   └───1
|   |   |   └───...
|   └───filelists
|   |   └───0
|   |   |   |   all.txt (caffe-style filelist)
|   |   └───1
|   |   └───...
|   └───testset_ratio_0.3
|   |   └───split_0
|   |   |   └───0
|   |   |   |   |   train.txt (caffe-style filelist)
|   |   |   |   |   test.txt (caffe-style filelist)
|   |   |   |   |   train_indices.json (python list of train sample indices)
|   |   |   |   |   test_indices.json (python list of test sample indices)
|   |   |   └───1
|   |   |   └───...
|   |   └───split_1
|   |   |   └───... (same as split_0)
|   |   └───split_2
|   |   |   └───... (same as split_0)
|   |   └───split_3
|   |   |   └───... (same as split_0)
|   |   └───split_4
|   |   |   └───... (same as split_0)
```


# Citation
If this work is useful for your research, please cite our paper:
```
@inproceedings{lin2021clear,
  title={The CLEAR Benchmark: Continual LEArning on Real-World Imagery},
  author={Lin, Zhiqiu and Shi, Jia and Pathak, Deepak and Ramanan, Deva},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2021}
}
```

# License
This code is freely available for free non-commercial use, and may be redistributed under these conditions. 
Third-party datasets and softwares are subject to their respective licenses. 