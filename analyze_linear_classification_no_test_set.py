# TODO: Fix the entire script after implementing new large_scale_feature_chunks.py

# 100 per bucket: 
from analyze_linear_classification import NEGATIVE_LABEL_SETS, get_negative_dataset_folder_paths, only_positive_accuracy, avg_per_class_accuracy
    # python analyze_linear_classification_no_test_set.py --train_mode nearest_mean --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode nearest_mean_normalized --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_wd --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_normalized_weight --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_normalized_projected --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_normalized_projected_true --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_normalized_feature --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_normalized_both --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode mlp --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    
    
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_scratch --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_imgnet --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_moco --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_byol --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

    # TODO
    # CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode moco_v2_imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode byol_imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    
# 300 per bucket:
    # python analyze_linear_classification_no_test_set.py --train_mode nearest_mean_normalized --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_wd --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_normalized_projected --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode linear_normalized_projected_true --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_scratch --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_imgnet --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_moco --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification_no_test_set.py --train_mode cnn_byol --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

    # New TODO
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode moco_v2_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode byol_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_no_test_set.py --only_label_set dynamic --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
  
# Reverse
    # 100 per bucket
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode linear_wd --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode linear_normalized_projected_true --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode nearest_mean_normalized --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_scratch --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_imgnet --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_byol --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_moco --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode moco_v2_imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode byol_imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    
    # 300 per bucket
        # python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode linear_wd --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode linear_normalized_projected_true --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode nearest_mean_normalized --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_scratch --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_imgnet --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_byol --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_moco --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode moco_v2_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode byol_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification_no_test_set.py --reverse_order --only_label_set dynamic --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# Negative 
#  300+3000 negative per bucket
        # python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode nearest_mean_normalized --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode cnn_scratch --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode cnn_imgnet --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=6 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode cnn_byol --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=7 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode cnn_moco --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=8 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode moco_v2_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode byol_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification_no_test_set.py --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar


import sys
sys.path.append("./CLIP")
import os
import clip
import torch
import faiss_utils
from faiss_utils import KNearestFaissFeatureChunks
import numpy as np
import time
import copy
from tqdm import tqdm
from datetime import datetime

# from large_scale_feature import argparser, get_clip_loader, get_clip_features, get_feature_name, FlickrAccessor, FlickrFolder, get_flickr_accessor
from analyze_feature_variation import argparser, get_dataset_folder_paths
import large_scale_feature_chunks
import argparse
import random
import importlib
from prepare_clip_dataset import QUERY_TITLE_DICT, LABEL_SETS
from utils import divide, normalize, load_pickle, save_obj_as_pickle
from training_utils import CLIPDataset
import training_utils

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_SET_RATIO = 0.3
TRAIN_SET_RATIO = 0.7
from analyze_linear_classification import HyperParameter, argparser, ALL_TRAIN_MODES, TRAIN_MODES_CATEGORY, HYPER_DICT, get_all_query, make_numpy_loader, make_image_loader, get_loaders_from_dataset_dict
from analyze_linear_classification import get_exclusive_loaders_from_dataset_dict, get_excluded_bucket_idx, make_model, make_optimizer, make_scheduler, test, get_input_size
from analyze_linear_classification import remove_random_crop_from_loader
from analyze_linear_classification import NEGATIVE_LABEL, NEGATIVE_LABEL_SETS, get_negative_dataset_folder_paths

def dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO):
    return "_".join(['train', str(TRAIN_SET_RATIO), 'test', str(TEST_SET_RATIO)])

def split_dataset(query_dict, TRAIN_SET_RATIO=TRAIN_SET_RATIO, TEST_SET_RATIO=TEST_SET_RATIO):
    all_query = get_all_query(query_dict)
    
    dataset_dict = {}
    num_of_data = len(query_dict[all_query[0]]['metadata'])
    for query in all_query:
        assert num_of_data == len(query_dict[query]['metadata'])
    
    data_indices = list(range(num_of_data))
    random.shuffle(data_indices)
    
    test_set_size = int(TEST_SET_RATIO * num_of_data)
    train_set_size = int(TRAIN_SET_RATIO * num_of_data)
    if not train_set_size + test_set_size == num_of_data:
        import pdb; pdb.set_trace()
        
    test_set_indices = data_indices[:test_set_size]
    train_set_indices = data_indices[test_set_size:]

    def gather_data(query, indices):
        return {
            'clip_features': [query_dict[query]['clip_features'][i] for i in indices],
            'metadata': [query_dict[query]['metadata'][i] for i in indices],
            'D': [query_dict[query]['D'][i] for i in indices],
        }
    for query in all_query:
        dataset_dict[query] = {}
        
        dataset_dict[query]['train_set'] = gather_data(query, train_set_indices)
        dataset_dict[query]['test_set'] = gather_data(query, test_set_indices)
        dataset_dict[query]['all'] = gather_data(query, data_indices)

    # For negative
    if NEGATIVE_LABEL in query_dict:
        dataset_dict[NEGATIVE_LABEL] = {}
        num_of_data = len(query_dict[NEGATIVE_LABEL]['metadata'])
        data_indices = list(range(num_of_data))
        random.shuffle(data_indices)

        test_set_size = int(TEST_SET_RATIO * num_of_data)
        train_set_size = int(TRAIN_SET_RATIO * num_of_data)
            
        test_set_indices = data_indices[:test_set_size]
        train_set_indices = data_indices[test_set_size:]

        dataset_dict[NEGATIVE_LABEL]['train_set'] = gather_data(NEGATIVE_LABEL, train_set_indices)
        dataset_dict[NEGATIVE_LABEL]['test_set'] = gather_data(NEGATIVE_LABEL, test_set_indices)
        dataset_dict[NEGATIVE_LABEL]['all'] = gather_data(NEGATIVE_LABEL, data_indices)
    #
    return dataset_dict

def get_all_loaders_from_dataset_dict(all_dataset_dict, hyperparameter, excluded_bucket_idx=0):
    # Return len(all_dataset_dict) - 1 loaders, each with all bucket except the current one
    all_bucket = sorted(list(all_dataset_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the loaders")
    dataset_dict = {k: all_dataset_dict[k] for k in all_bucket if k != excluded_bucket_idx}
    loaders_dict = {}

    all_query = sorted(list(dataset_dict[list(dataset_dict.keys())[0]].keys()))
    for feature_name in ['clip_features', 'metadata']:
        loaders_dict[feature_name] = {}
        for k_name in ['train_set', 'test_set']:
            items = []
            for b_idx in list(dataset_dict.keys()):
                for q_idx, query in enumerate(all_query):
                    items += [(f, q_idx)
                              for f in dataset_dict[b_idx][query][k_name][feature_name]]
            if k_name == 'train_set':
                shuffle = True
            else:
                shuffle = False
            if feature_name == 'metadata':
                loader = make_image_loader(items, hyperparameter, shuffle=shuffle)
            elif feature_name in ['clip_features']:
                loader = make_numpy_loader(items, hyperparameter, shuffle=shuffle)
            loaders_dict[feature_name][k_name] = loader
    return loaders_dict    

def train(train_loader, test_loader, 
          train_mode, input_size, output_size,
          epochs=150, lr=0.1, weight_decay=1e-5, step_size=60):
    if train_mode in TRAIN_MODES_CATEGORY['nearest_mean']:
        network = {}
        loaders = {'train' : train_loader, 'test' : test_loader}
        for batch, data in enumerate(train_loader):
            inputs, labels = data
            if train_mode in TRAIN_MODES_CATEGORY['feature_normalized']:
                inputs = inputs/torch.linalg.norm(inputs, dim=1, keepdim=True)
            for i in range(inputs.shape[0]):
                input_i = inputs[i]
                label_i = int(labels[i])
                if label_i in network:
                    network[label_i] = input_i.cpu().numpy() + network[label_i]
                else:
                    network[label_i] = input_i.cpu().numpy()
        for label_i in network:
            network[label_i] = network[label_i] / np.linalg.norm(network[label_i])
        
        return network, {'train': test(train_loader, network, train_mode),
                         'test': test(test_loader, network, train_mode)}
    else:
        network = make_model(train_mode, input_size, output_size).cuda()
        optimizer = make_optimizer(network, lr, weight_decay)
        scheduler = make_scheduler(optimizer, step_size=step_size)
        criterion = torch.nn.NLLLoss(reduction='mean')

        avg_loss_per_epoch = []
        avg_acc_per_epoch = []

        avg_test_acc_per_epoch = []

        best_test_epoch_train_acc = 0
        best_test_acc = 0
        best_test_epoch = None

        best_test_network = None

        loaders = {'train' : train_loader, 'test' : test_loader}
        for epoch in range(0, epochs):
            print(f"Epoch {epoch}")
            for phase in loaders.keys():
                # import pdb; pdb.set_trace()
                if phase == 'train':
                    if train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
                        print("Using Frozen Network in Eval Mode")
                        network.eval()
                    else:
                        network.train()
                else:
                    network.eval()

                running_loss = 0.0
                running_corrects = 0.
                count = 0

                pbar = loaders[phase]

                for batch, data in enumerate(pbar):
                    inputs, labels = data
                    count += inputs.size(0)
                        
                    inputs = inputs.cuda()
                    if train_mode in TRAIN_MODES_CATEGORY['feature_normalized']:
                        inputs = inputs/torch.linalg.norm(inputs, dim=1, keepdim=True)
                    labels = labels.cuda()

                    if phase == 'train': optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = network(inputs)
                        _, preds = torch.max(outputs, 1)

                        log_probability = torch.nn.functional.log_softmax(outputs, dim=1)
                        loss = criterion(log_probability, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            if train_mode in TRAIN_MODES_CATEGORY['projected_normalized']:
                                network.weight.data = network.weight.data/torch.linalg.norm(network.weight.data, dim=1, keepdim=True)
                            elif train_mode in TRAIN_MODES_CATEGORY['projected_normalized_true']:
                                msk = torch.linalg.norm(network.weight.data, dim=1) < 1.
                                network.weight.data[msk] = (network.weight.data/torch.linalg.norm(network.weight.data, dim=1, keepdim=True))[msk]

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                avg_loss = float(running_loss)/count
                avg_acc = float(running_corrects)/count
                if phase == 'train': 
                    avg_loss_per_epoch.append(avg_loss)
                    avg_acc_per_epoch.append(avg_acc)
                    scheduler.step()
                elif phase == 'test':
                    avg_test_acc_per_epoch.append(avg_acc)
                    if avg_acc > best_test_acc:
                        print(f"Best test accuracy at epoch {epoch} being {avg_acc}")
                        best_test_epoch = epoch
                        best_test_acc = avg_acc
                        best_test_epoch_train_acc = avg_acc_per_epoch[-1]
                        print(f"Train accuracy at epoch {epoch} being {best_test_epoch_train_acc}")
                        best_test_network = copy.deepcopy(network.state_dict())
                print(f"Epoch {epoch}: Average {phase} Loss {avg_loss}, Accuracy {avg_acc:.2%}")
            print()
        print(
            f"Best Test Accuracy (for best test model): {avg_test_acc_per_epoch[best_test_epoch]:.2%}")
        print(f"Best Test Accuracy overall: {max(avg_test_acc_per_epoch):.2%}")
        network.load_state_dict(best_test_network)
        test_acc = test(test_loader, network, train_mode, save_loc=None)
        return network, {'train': best_test_epoch_train_acc, 'test': best_test_acc}


if __name__ == '__main__':
    args = argparser.parse_args()
    if not args.avoid_multiple_class and not args.use_negative_samples:
        import pdb; pdb.set_trace()
        
    start = time.time()
    
    if args.use_negative_samples:
        print("Use Negative samples")
        dataset_paths_dict = get_negative_dataset_folder_paths(args.folder_path, args.num_of_bucket)
    else:
        dataset_paths_dict = get_dataset_folder_paths(
            args.folder_path, args.num_of_bucket, args.query_title, 
            args.class_size, args.avoid_multiple_class, reverse_order=args.reverse_order, nn_size=args.nn_size)
    bucket_dict = {}
    
    excluded_bucket_idx = get_excluded_bucket_idx(args.moco_model)

    for label_set in dataset_paths_dict.keys():
        if label_set in args.excluded_label_set:
            print(f"<<<<<<<<<<<<<<<<<<<<<<<<<Skipping label set {label_set}")
            print()
            print()
            continue
        if args.only_label_set:
            if label_set == args.only_label_set:
                print(f"Only do label set {args.only_label_set}")
            else:
                print(f"<<<<<<<<<<<<<<<<<<<<<<<<<Skipping label set {label_set}")
                print()
                print()
                continue
        main_label_set_path = dataset_paths_dict[label_set]['main_label_set_path']
        sub_folder_paths = dataset_paths_dict[label_set]['sub_folder_paths']
        if args.use_negative_samples:
            print("Not checking the info dict..")
        else:
            info_dict = {
                'query_title': args.query_title,
                'query_title_name': QUERY_TITLE_DICT[args.query_title],
                'model_name': args.model_name,
                'folder_path': args.folder_path,
                'clip_dataset_paths': dataset_paths_dict[label_set],
                'label_set': label_set,
                'class_size': args.class_size,
                'avoid_multiple_class': args.avoid_multiple_class,
                'nn_size': args.nn_size,
                'num_of_bucket' : args.num_of_bucket,
                'moco_model': args.moco_model,
                'arch' : args.arch,
            }
            info_dict_path = os.path.join(main_label_set_path, "info_dict.pickle")
            if os.path.exists(info_dict_path):
                saved_info_dict = load_pickle(info_dict_path)
                if not saved_info_dict == info_dict:
                    print("Info dict does not align")
                    import pdb; pdb.set_trace()
            else:
                print("No info dict was saved")
                import pdb; pdb.set_trace()
        
        print(f"Processing {label_set}.. ")
        
        query_dict = {} # Saved the query results for each data bucket

        query_dict_path = os.path.join(main_label_set_path, "query_dict.pickle")
        if not os.path.exists(query_dict_path):
            print(f"Query dict does not exist for {label_set}")
            import pdb; pdb.set_trace()
            continue
        query_dict = load_pickle(query_dict_path)
        queries = list(query_dict[0].keys())
        print(queries)
        print(f"We have {len(queries)} queries.")
        
        dataset_dict_path = os.path.join(main_label_set_path,
                                         f"dataset_dict_{dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO)}.pickle")
        loaders_dict_path = os.path.join(main_label_set_path,
                                         f"loaders_dict_{dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO)}_{args.train_mode}.pickle")
        exclusive_loaders_dict_path = os.path.join(main_label_set_path,
                                                   f"exclusive_loaders_dict_{dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO)}_{args.train_mode}_ex_{excluded_bucket_idx}.pickle")
        all_loaders_dict_path = os.path.join(main_label_set_path,
                                             f"all_loaders_dict_{dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO)}_{args.train_mode}_ex_{excluded_bucket_idx}.pickle")
        if os.path.exists(dataset_dict_path):
            print(f"{dataset_dict_path} already exists.")
            dataset_dict = load_pickle(dataset_dict_path)
        else:
            dataset_dict = {}  # Saved the splitted dataset for each bucket

            for b_idx, sub_folder_path in enumerate(sub_folder_paths):
                dataset_dict_i_path = os.path.join(sub_folder_path,
                                                f"dataset_dict_{b_idx}_{dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO)}.pickle")
                if not os.path.exists(dataset_dict_i_path):
                    print(f"<<<<<<<<<<<First split the dataset for bucket {b_idx}")
                    dataset_dict[b_idx] = split_dataset(query_dict[b_idx])
                    save_obj_as_pickle(dataset_dict_i_path, dataset_dict[b_idx])
                else:
                    print(f"Load from {dataset_dict_i_path}")
                    dataset_dict[b_idx] = load_pickle(dataset_dict_i_path)
                
            save_obj_as_pickle(dataset_dict_path, dataset_dict)
        
        if os.path.exists(loaders_dict_path) and os.path.exists(exclusive_loaders_dict_path) and os.path.exists(all_loaders_dict_path):
            print(f"{loaders_dict_path} already exists.")
            loaders_dict = load_pickle(loaders_dict_path)
            exclusive_loaders_dict = load_pickle(exclusive_loaders_dict_path)
            all_loaders_dict = load_pickle(all_loaders_dict_path)
        else:
            loaders_dict = {}  # Saved the splitted loader for each bucket

            for b_idx, sub_folder_path in enumerate(sub_folder_paths):
                loaders_dict_i_path = os.path.join(sub_folder_path,
                                                f"loaders_dict_{b_idx}_{dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO)}_{args.train_mode}.pickle")
                if not os.path.exists(loaders_dict_i_path):
                    print(f"<<<<<<<<<<<First create the dataset loader for bucket {b_idx}")
                    loaders_dict[b_idx] = get_loaders_from_dataset_dict(
                        dataset_dict[b_idx], HYPER_DICT[args.train_mode])
                    save_obj_as_pickle(loaders_dict_i_path, loaders_dict[b_idx])
                else:
                    print(f"Load from {loaders_dict_i_path}")
                    loaders_dict[b_idx] = load_pickle(loaders_dict_i_path)
                
            exclusive_loaders_dict = get_exclusive_loaders_from_dataset_dict(dataset_dict, HYPER_DICT[args.train_mode], excluded_bucket_idx=excluded_bucket_idx)
            all_loaders_dict = get_all_loaders_from_dataset_dict(dataset_dict, HYPER_DICT[args.train_mode], excluded_bucket_idx=excluded_bucket_idx)
            save_obj_as_pickle(exclusive_loaders_dict_path, exclusive_loaders_dict)    
            save_obj_as_pickle(loaders_dict_path, loaders_dict)
            save_obj_as_pickle(all_loaders_dict_path, all_loaders_dict)
        
        results_dict_path = os.path.join(main_label_set_path,
                                         f"results_dict_{dataset_str(TRAIN_SET_RATIO, TEST_SET_RATIO)}_{args.train_mode}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_path) or True:
            all_query = sorted(list(dataset_dict[0].keys()))
            models_dict = {'models': {}, 'b1_b2_accuracy_matrix': {}, 'accuracy': {},
                           'b1_b2_per_class_accuracy_dict': {},
                           'only_positive_accuracy_test': {},
                           'avg_per_class_accuracy_test': {}}
            all_models_dict = {'models': {}, 'accuracy_matrix': {},
                               'per_class_accuracy_dict': {},
                               'only_positive_accuracy_test': {},
                               'avg_per_class_accuracy_test': {}}
            results_dict = {'single' : models_dict, 'all' : all_models_dict}
            all_bucket = len(list(loaders_dict.keys()))
            if args.train_mode in TRAIN_MODES_CATEGORY['cnn'] + TRAIN_MODES_CATEGORY['cnn_linear_feature']:
                feature_name_list = ['metadata']
            elif args.train_mode in TRAIN_MODES_CATEGORY['linear'] + TRAIN_MODES_CATEGORY['nearest_mean']:
                feature_name_list = ['clip_features']
            for feature_name in feature_name_list:
                continue
                single_accuracy_test = np.zeros((all_bucket, all_bucket))
                only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
                avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
                models_dict['models'][feature_name] = {}
                models_dict['accuracy'][feature_name] = {}
                b1_b2_per_class_accuracy_dict = {}
                input_size = get_input_size(feature_name)
                for b1 in range(all_bucket):
                    b1_b2_per_class_accuracy_dict[b1] = {}
                    train_loader = loaders_dict[b1][feature_name]['train_set']
                    if args.train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
                        train_loader = remove_random_crop_from_loader(train_loader)
                    test_loader = loaders_dict[b1][feature_name]['test_set']
                    single_model, single_accuracy_b1 = train(train_loader, test_loader, args.train_mode,
                                                             input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
                                                             lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay, step_size=HYPER_DICT[args.train_mode].step)
                    models_dict['models'][feature_name][b1] = None
                    models_dict['accuracy'][feature_name][b1] = single_accuracy_b1
                    for b2 in range(all_bucket):
                        test_loader_b2 = loaders_dict[b2][feature_name]['test_set']
                        single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, args.train_mode, class_names=all_query)
                        b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
                        only_positive_accuracy_test[b1][b2] = only_positive_accuracy(per_class_accuracy_b1_b2)
                        avg_per_class_accuracy_test[b1][b2] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
                        print(
                            f"Train {b1}, test on {b2}: {single_accuracy_b1_b2}")
                        single_accuracy_test[b1][b2] = single_accuracy_b1_b2
                models_dict['b1_b2_accuracy_matrix'][feature_name] = single_accuracy_test
                models_dict['b1_b2_per_class_accuracy_dict'][feature_name] = b1_b2_per_class_accuracy_dict
                models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy_test
                models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy_test
            
            for feature_name in feature_name_list:
                print(f"{feature_name}:")
                train_loader = all_loaders_dict[feature_name]['train_set']
                if args.train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
                    train_loader = remove_random_crop_from_loader(train_loader)
                test_loader = all_loaders_dict[feature_name]['test_set']
                input_size = get_input_size(feature_name)
                all_model, all_accuracy = train(train_loader, test_loader, args.train_mode, 
                                                input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
                                                lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay, step_size=HYPER_DICT[args.train_mode].step)
                print(all_accuracy)
                all_models_dict['accuracy_matrix'][feature_name] = all_accuracy
                all_models_dict['models'][feature_name] = all_model
                test_accuracy_all, per_class_accuracy_all = test(test_loader, all_model, args.train_mode, class_names=all_query)
                all_models_dict['per_class_accuracy_dict'][feature_name] = per_class_accuracy_all
                all_models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy(per_class_accuracy_all)
                all_models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy(per_class_accuracy_all)
                print(f"Baseline: {test_accuracy_all:.4%} (per sample), {all_models_dict['only_positive_accuracy_test'][feature_name]:.4%} (pos only), {all_models_dict['avg_per_class_accuracy_test'][feature_name]:.4%} (per class avg)")
            save_obj_as_pickle(results_dict_path, results_dict)
            print(f"Saved at {results_dict_path}")
        
        assert os.path.exists(results_dict_path)
        results_dict = load_pickle(results_dict_path)
        all_query = sorted(list(dataset_dict[0].keys()))
        all_models_dict = results_dict['all']
        models_dict = results_dict['single']
        all_bucket = len(list(loaders_dict.keys()))
        if args.train_mode in TRAIN_MODES_CATEGORY['cnn'] + TRAIN_MODES_CATEGORY['cnn_linear_feature']:
            feature_name_list = ['metadata']
        elif args.train_mode in TRAIN_MODES_CATEGORY['linear'] + TRAIN_MODES_CATEGORY['nearest_mean']:
            feature_name_list = ['clip_features']
        
        if not 'b1_b2_per_class_accuracy_dict' in models_dict:
            models_dict['b1_b2_per_class_accuracy_dict'] = {}
            models_dict['only_positive_accuracy_test'] = {}
            models_dict['avg_per_class_accuracy_test'] = {}
            for feature_name in feature_name_list:
                only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
                avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
                b1_b2_per_class_accuracy_dict = {}
                for b1 in range(all_bucket):
                    b1_b2_per_class_accuracy_dict[b1] = {}
                    # train_loader = loaders_dict[b1][feature_name]['train_set']
                    # if args.train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
                    #     train_loader = remove_random_crop_from_loader(train_loader)
                    # val_loader = loaders_dict[b1][feature_name]['val_set']
                    test_loader = loaders_dict[b1][feature_name]['test_set']
                    # single_model, single_accuracy_b1 = train(train_loader, val_loader, test_loader, args.train_mode,
                    #                                          input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
                    #                                          lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay, step_size=HYPER_DICT[args.train_mode].step)
                    single_model = models_dict['models'][feature_name][b1]
                    # models_dict['accuracy'][feature_name][b1] = single_accuracy_b1
                    for b2 in range(all_bucket):
                        test_loader_b2 = loaders_dict[b2][feature_name]['test_set']
                        single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, args.train_mode, class_names=all_query)
                        b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
                        only_positive_accuracy_test[b1][b2] = only_positive_accuracy(per_class_accuracy_b1_b2)
                        avg_per_class_accuracy_test[b1][b2] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
                        print(
                            f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1][b2]:.4%} (pos only), {avg_per_class_accuracy_test[b1][b2]:.4%} (per class avg)")
                        # single_accuracy_test[b1][b2] = single_accuracy_b1_b2
                # models_dict['b1_b2_accuracy_matrix'][feature_name] = single_accuracy_test
                models_dict['b1_b2_per_class_accuracy_dict'][feature_name] = b1_b2_per_class_accuracy_dict
                models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy_test
                models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy_test
            save_obj_as_pickle(results_dict_path, results_dict)
            print(f"Re-Saved at {results_dict_path}")
        
        if not 'per_class_accuracy_dict' in all_models_dict:
            all_models_dict['per_class_accuracy_dict'] = {}
            all_models_dict['only_positive_accuracy_test'] = {}
            all_models_dict['avg_per_class_accuracy_test'] = {}
            for feature_name in feature_name_list:
                print(f"{feature_name}:")
                # train_loader = all_loaders_dict[feature_name]['train_set']
                # if args.train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
                #     train_loader = remove_random_crop_from_loader(train_loader)
                # val_loader = all_loaders_dict[feature_name]['val_set']
                test_loader = all_loaders_dict[feature_name]['test_set']
                # input_size = get_input_size(feature_name)
                # all_model, all_accuracy = train(train_loader, val_loader, test_loader, args.train_mode, 
                #                                 input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
                #                                 lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay, step_size=HYPER_DICT[args.train_mode].step)
                all_model = all_models_dict['models'][feature_name]
                test_accuracy_all, per_class_accuracy_all = test(test_loader, all_model, args.train_mode, class_names=all_query)
                # print(all_accuracy)
                # all_models_dict['accuracy_matrix'][feature_name] = all_accuracy
                all_models_dict['per_class_accuracy_dict'][feature_name] = per_class_accuracy_all
                all_models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy(per_class_accuracy_all)
                all_models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy(per_class_accuracy_all)

            save_obj_as_pickle(results_dict_path, results_dict)
            print(f"Saved at {results_dict_path}")
