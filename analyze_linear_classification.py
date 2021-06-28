# TODO: Fix the entire script after implementing new large_scale_feature_chunks.py

# Run 1-6: python analyze_linear_classification.py --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_wd --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_normalized_weight --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_normalized_projected --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_normalized_projected_true --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_normalized_feature --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_normalized_both --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode mlp --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode nearest_mean --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode nearest_mean_normalized --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_scratch --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_imgnet --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_moco --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_byol --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # New on 1-1
    # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --only_label_set dynamic --train_mode moco_v2_imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --only_label_set dynamic --train_mode byol_imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --only_label_set dynamic --train_mode imgnet_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --only_label_set dynamic --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_github --model_name RN50 --class_size 100 --nn_size 2048 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# 300 per bucket positive only:
    # python analyze_linear_classification.py --train_mode linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_wd --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_normalized_projected --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode linear_normalized_projected_true --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode nearest_mean_normalized --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_scratch --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_imgnet --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_moco --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --train_mode cnn_byol --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # New on 1-1
    # python analyze_linear_classification.py --only_label_set dynamic --train_mode moco_v2_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --only_label_set dynamic --train_mode byol_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --only_label_set dynamic --train_mode imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --only_label_set dynamic --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
 # 300 per bucket positive only no test set:
    # python analyze_linear_classification.py --mode no_test_set --train_mode nearest_mean_normalized --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --mode no_test_set --train_mode linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --mode no_test_set --train_mode linear_wd --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --mode no_test_set --train_mode linear_normalized_projected --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --mode no_test_set --train_mode linear_normalized_projected_true --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    
    # python analyze_linear_classification.py --mode no_test_set --train_mode cnn_scratch --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --mode no_test_set --train_mode cnn_imgnet --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --mode no_test_set --train_mode cnn_moco --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # python analyze_linear_classification.py --mode no_test_set --train_mode cnn_byol --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

    # New TODO
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification.py --mode no_test_set --only_label_set dynamic --train_mode moco_v2_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification.py --mode no_test_set --only_label_set dynamic --train_mode byol_imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification.py --mode no_test_set --only_label_set dynamic --train_mode imgnet_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification.py --mode no_test_set --only_label_set dynamic --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification.py --mode no_test_set --only_label_set dynamic --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --model_name RN50 --class_size 300 --nn_size 8000 --avoid_multiple_class --query_title none --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --model_name RN50 --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
  
# Negative 
#  300+3000 negative per bucket
        # python analyze_linear_classification.py --use_negative_samples --train_mode nearest_mean_normalized --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode cnn_scratch --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode cnn_imgnet --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode cnn_byol --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode cnn_moco --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode moco_v2_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode byol_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # python analyze_linear_classification.py --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# Negative no test set
#  300+3000 negative per bucket
        # python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode nearest_mean_normalized --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode cnn_scratch --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=5 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode cnn_imgnet --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=6 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode cnn_byol --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=7 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode cnn_moco --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=8 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0 --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode moco_v2_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode byol_imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode imgnet_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

# MLP
    # Negative no test set
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode moco_v2_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode byol_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification.py --mode no_test_set --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
    # Negative with test set
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode default --use_negative_samples --train_mode mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=1 python analyze_linear_classification.py --mode default --use_negative_samples --train_mode moco_v2_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=2 python analyze_linear_classification.py --mode default --use_negative_samples --train_mode byol_imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=3 python analyze_linear_classification.py --mode default --use_negative_samples --train_mode imgnet_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar
        # CUDA_VISIBLE_DEVICES=4 python analyze_linear_classification.py --mode default --use_negative_samples --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp --folder_path /scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18/ --num_of_bucket 11 --moco_model /project_data/ramanan/zhiqiu/yfcc_moco_models/feb_18_bucket_11_idx_0_gpu_8/checkpoint_0199.pth.tar

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

from analyze_feature_variation_negative import NEGATIVE_LABEL, NEGATIVE_LABEL_SETS, get_negative_dataset_folder_paths

device = "cuda" if torch.cuda.is_available() else "cpu"

MODE_DICT = {
    'default' : {
        'VAL_SET_RATIO' : 0.1,
        'TEST_SET_RATIO' : 0.1,
        'TRAIN_SET_RATIO' : 0.8,
    },
    'no_test_set': {
        'TEST_SET_RATIO': 0.3,
        'TRAIN_SET_RATIO': 0.7,
    },
}

class HyperParameter():
    def __init__(self, network_name, epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.):
        self.network_name = network_name
        self.epochs = epochs
        self.step = step
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
    
    def get_detail_str(self):
        if self.network_name in TRAIN_MODES_CATEGORY['nearest_mean']:
            return self.network_name
        else:
            return "_".join([self.network_name, 'ep', self.epochs, 'step', self.step, 'b', self.batch_size, 'lr', self.lr, 'wd', self.weight_decay])


# argparser = argparse.ArgumentParser()
# Below are in large_scale_feature_chunks already
# argparser.add_argument("--model_name", 
#                         default='RN50', choices=clip.available_models(),
#                         help="The CLIP model to use")
# argparser.add_argument("--folder_path", 
#                         default='/scratch/zhiqiu/yfcc100m_all/images_minbyte_10_valid_uploaded_date_feb_18',
#                         help="The folder with all the computed+normalized CLIP + Moco features got by large_scale_features_chunk.py")
# argparser.add_argument('--num_of_bucket', default=11, type=int,
#                        help='number of bucket')
# argparser.add_argument("--moco_model",
#                        default='',
#                        help="The moco model to use")
# argparser.add_argument('--arch', metavar='ARCH', default='resnet50',
#                        help='model architecture: ' +
#                        ' (default: resnet50)')
# argparser.add_argument("--query_title", 
#                         default='photo', 
#                         choices=QUERY_TITLE_DICT.keys(),
#                         help="The query title")
# argparser.add_argument('--class_size', default=2000, type=int,
#                        help='number of (max score) samples per class per bucket')
# argparser.add_argument("--avoid_multiple_class",
#                        action='store_true',
#                        help="(Must be true for linear classification) Only keep the max scoring images if set True")
# argparser.add_argument("--nn_size",
#                        default=2048, type=int,
#                        help="If avoid_multiple_class set to True, then first retrieve this number of top score images, and filter out duplicate")
argparser.add_argument("--excluded_label_set",
                       default=['tech_7', 'imagenet1K'],
                       help="Do not evaluate on these listed label set")
argparser.add_argument("--only_label_set",
                       default=None,
                       help="If set to a [labet_set], only evaluate on this label set")
argparser.add_argument("--use_negative_samples",
                       action='store_true',
                       help="If set True, then the dataset contains a negative class (run analyze_feature_variation_negative.py first)")

ALL_TRAIN_MODES = ['mlp', 'linear', 'linear_wd', 'nearest_mean', 'linear_normalized_weight', 'linear_normalized_feature', 'linear_normalized_projected', 'linear_normalized_projected_true',
                   'linear_normalized_both', 'nearest_mean_normalized', 'cnn_scratch', 'cnn_imgnet', 'cnn_moco', 'cnn_byol',
                   'moco_v2_imgnet_linear', 'byol_imgnet_linear', 'imgnet_linear', 'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear', 'cnn_moco_yfcc_feb18_gpu_8_bucket_0',
                   'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_github',
                   'moco_v2_imgnet_mlp', 'byol_imgnet_mlp', 'imgnet_mlp', 'moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp']

TRAIN_MODES_CATEGORY = {
    'cnn': ['cnn_scratch', 'cnn_imgnet', 'cnn_moco', 'cnn_byol', 'cnn_moco_yfcc_feb18_gpu_8_bucket_0'],
    'projected_normalized': ['linear_normalized_projected'], # Compare to weight_normalize with only normalize at test time, this is more aggressive and normalize right after each SGD update.
    'projected_normalized_true': ['linear_normalized_projected_true'], # Compare to projected_normalized, this only reduce the column with norm > 1 (so this is in fact the true PDG algorithm).
    'feature_normalized': ['nearest_mean_normalized', 'linear_normalized_projected', 'linear_normalized_projected_true', 'linear_normalized_both', 'linear_normalized_feature', 'nearest_mean_normalized'], # L2 normalize feature for both train and test time.
    'weight_normalized': ['linear_normalized_weight', 'linear_normalized_both'], # L2 normalize weight (each column) only during test time
    'linear': ['linear_normalized_projected_true', 'linear_normalized_projected', 'linear_normalized_both', 'linear_normalized_feature', 'nearest_mean_normalized', 'linear', 'linear_wd', 'linear_normalized_weight'],
    'mlp' : ['mlp'],
    'nearest_mean': ['nearest_mean', 'nearest_mean_normalized'],
    'cnn_linear_feature': ['moco_v2_imgnet_linear', 'byol_imgnet_linear', 'imgnet_linear', 'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear', 'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_github'],
    'cnn_mlp_feature': ['moco_v2_imgnet_mlp', 'byol_imgnet_mlp', 'imgnet_mlp', 'moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp']
}

HYPER_DICT = {
    'mlp': HyperParameter('mlp', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'moco_v2_imgnet_mlp': HyperParameter('moco_v2_imgnet_mlp', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'byol_imgnet_mlp': HyperParameter('byol_imgnet_mlp', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'imgnet_mlp': HyperParameter('imgnet_mlp', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp': HyperParameter('moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'cnn_scratch': HyperParameter('cnn_scratch', epochs=150, step=60, batch_size=64, lr=0.1, weight_decay=1e-5),
    'cnn_imgnet': HyperParameter('cnn_imgnet', epochs=150, step=60, batch_size=64, lr=0.1, weight_decay=1e-5),
    'cnn_moco': HyperParameter('cnn_moco', epochs=150, step=60, batch_size=64, lr=0.1, weight_decay=1e-5),
    'cnn_byol': HyperParameter('cnn_byol', epochs=150, step=60, batch_size=64, lr=0.1, weight_decay=1e-5),
    'cnn_moco_yfcc_feb18_gpu_8_bucket_0': HyperParameter('cnn_moco_yfcc_feb18_gpu_8_bucket_0', epochs=150, step=60, batch_size=64, lr=0.1, weight_decay=1e-5),
    'linear': HyperParameter('linear', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'moco_v2_imgnet_linear': HyperParameter('moco_v2_imgnet_linear', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'byol_imgnet_linear': HyperParameter('byol_imgnet_linear', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'imgnet_linear': HyperParameter('imgnet_linear', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear': HyperParameter('moco_v2_yfcc_feb18_bucket_0_gpu_8_linear', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_github': HyperParameter('moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_github', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0.),
    'linear_wd': HyperParameter('linear_wd', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=1e-5),
    'linear_normalized_weight': HyperParameter('linear_normalized_weight', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0),
    'linear_normalized_feature': HyperParameter('linear_normalized_feature', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0),
    'linear_normalized_both': HyperParameter('linear_normalized_both', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0),
    'linear_normalized_projected': HyperParameter('linear_normalized_projected', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0),
    'linear_normalized_projected_true': HyperParameter('linear_normalized_projected_true', epochs=100, step=60, batch_size=256, lr=0.1, weight_decay=0),
    'nearest_mean': HyperParameter('nearest_mean', batch_size=256),
    'nearest_mean_normalized': HyperParameter('nearest_mean_normalized', batch_size=256),
}

argparser.add_argument('--train_mode',
                       default='linear', choices=ALL_TRAIN_MODES,
                       help='Train mode')
argparser.add_argument('--mode',
                       default='default', choices=MODE_DICT.keys(),
                       help='Mode for dataset split')


def to_remove_random_crop(train_mode):
    # Return true if need to manually remove random crop from train loaders
    # TODO: Check it against newly implemented train_mode
    return train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature'] + TRAIN_MODES_CATEGORY['cnn_mlp_feature']

def use_val_set(mode):
    return 'VAL_SET_RATIO' in MODE_DICT[mode]

def dataset_str(mode):
    if use_val_set(mode):
        return "_".join(['train', str(MODE_DICT[mode]['TRAIN_SET_RATIO']),
                         'val', str(MODE_DICT[mode]['VAL_SET_RATIO']),
                         'test', str(MODE_DICT[mode]['TEST_SET_RATIO'])])
    else:
        return "_".join(['train', str(MODE_DICT[mode]['TRAIN_SET_RATIO']),
                         'test', str(MODE_DICT[mode]['TEST_SET_RATIO'])])


def get_all_query(query_dict):
    all_query = sorted(list(query_dict.keys()))
    # For negative
    if NEGATIVE_LABEL in all_query:
        print(f"Removed {NEGATIVE_LABEL}")
        all_query.remove(NEGATIVE_LABEL)
    #
    return all_query

def remove_random_crop_from_loader(loader):
    _, test_transform = training_utils.get_imgnet_transforms()
    loader.__dict__['dataset'].__dict__['transform'] = test_transform
    return loader

def split_dataset(query_dict, mode='default'):
    dataset_dict = {}
    
    def gather_data(query, indices):
        return {
            'clip_features': [query_dict[query]['clip_features'][i] for i in indices],
            'metadata': [query_dict[query]['metadata'][i] for i in indices],
            'D': [query_dict[query]['D'][i] for i in indices],
        }

    for query in query_dict:
        num_of_data = len(query_dict[query]['metadata'])
        # for query in all_query:
        #     assert num_of_data == len(query_dict[query]['metadata'])
        data_indices = list(range(num_of_data))
        random.shuffle(data_indices)
        if use_val_set(mode):
            val_set_size = int(MODE_DICT[mode]['VAL_SET_RATIO'] * num_of_data)
        else:
            val_set_size = 0
        val_set_indices = data_indices[:val_set_size]
        
        test_set_size = int(MODE_DICT[mode]['TEST_SET_RATIO'] * num_of_data)
        test_set_indices = data_indices[val_set_size:val_set_size+test_set_size]
        train_set_size = int(MODE_DICT[mode]['TRAIN_SET_RATIO'] * num_of_data)
        train_set_indices = data_indices[val_set_size+test_set_size:]
        total_size = sum(train_set_indices + val_set_indices + test_set_indices)
        if not total_size == num_of_data:
            import pdb; pdb.set_trace()
        dataset_dict[query] = {}
        dataset_dict[query]['train_set'] = gather_data(query, train_set_indices)
        if use_val_set(mode):
            dataset_dict[query]['val_set'] = gather_data(query, val_set_indices)
        dataset_dict[query]['test_set'] = gather_data(query, test_set_indices)
        # TODO: Handle when dataset_dict has empty val set
        dataset_dict[query]['all'] = gather_data(query, data_indices)

    return dataset_dict


def make_numpy_loader(items, hyperparameter, shuffle=False):
    return torch.utils.data.DataLoader(
        CLIPDataset(items),
        batch_size=hyperparameter.batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

def make_image_loader(items, hyperparameter, shuffle=False, fixed_crop=False):
    items = [(m.get_path(), l) for m, l in items]
    train_transform, test_transform = training_utils.get_imgnet_transforms()
    if shuffle and not fixed_crop:
        transform = train_transform
    else:
        transform = test_transform
    return training_utils.make_loader(
        items, transform, shuffle=shuffle, batch_size=hyperparameter.batch_size, num_workers=4
    )

def get_loaders_from_dataset_dict(dataset_dict, hyperparameter):
    all_query = sorted(list(dataset_dict.keys()))
    
    loaders_dict = {}
    
    for feature_name in ['clip_features', 'metadata']:
        loaders_dict[feature_name] = {}
        for k_name in dataset_dict[all_query[0]]:
            items = []
            for q_idx, query in enumerate(all_query):
                items += [(f, q_idx) for f in dataset_dict[query][k_name][feature_name]]
            if k_name == 'train_set':
                shuffle = True
            else:
                shuffle = False
            if feature_name == 'metadata':
                loader = make_image_loader(items, hyperparameter, shuffle=shuffle, fixed_crop=False)
            elif feature_name in ['clip_features']:
                loader = make_numpy_loader(items, hyperparameter, shuffle=shuffle)
            loaders_dict[feature_name][k_name] = loader
    return loaders_dict


def get_exclusive_loaders_from_dataset_dict(all_dataset_dict, hyperparameter, excluded_bucket_idx=0):
    # Return len(all_dataset_dict) - 1 loaders, each with all bucket except the current one
    all_bucket = sorted(list(all_dataset_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the loaders")
    dataset_dict = {k: all_dataset_dict[k] for k in all_bucket if k != excluded_bucket_idx}
    loaders_dict = {}

    all_query = sorted(list(dataset_dict[list(dataset_dict.keys())[0]].keys()))
    for b_idx in dataset_dict:
        loaders_dict[b_idx] = {}
        other_buckets = [b for b in dataset_dict.keys() if b != b_idx]
        for feature_name in ['clip_features', 'metadata']:
            loaders_dict[b_idx][feature_name] = {}
            for k_name in dataset_dict[b_idx][all_query[0]]:
                items = []
                for b_other_idx in other_buckets:
                    for q_idx, query in enumerate(all_query):
                        items += [(f, q_idx)
                                  for f in dataset_dict[b_other_idx][query][k_name][feature_name]]
                if k_name == 'train_set':
                    shuffle = True
                else:
                    shuffle = False
                if feature_name == 'metadata':
                    loader = make_image_loader(items, hyperparameter, shuffle=shuffle)
                elif feature_name in ['clip_features']:
                    loader = make_numpy_loader(items, hyperparameter, shuffle=shuffle)
                loaders_dict[b_idx][feature_name][k_name] = loader
    return loaders_dict

def get_all_loaders_from_dataset_dict(all_dataset_dict, hyperparameter, excluded_bucket_idx=0):
    all_bucket = sorted(list(all_dataset_dict.keys()))
    print(f"Excluding {excluded_bucket_idx} from the loaders")
    dataset_dict = {k: all_dataset_dict[k] for k in all_bucket if k != excluded_bucket_idx}
    loaders_dict = {}

    all_query = sorted(list(dataset_dict[list(dataset_dict.keys())[0]].keys()))
    for feature_name in ['clip_features', 'metadata']:
        loaders_dict[feature_name] = {}
        # for k_name in ['train_set', 'val_set', 'test_set']:
        for k_name in dataset_dict[list(dataset_dict.keys())[0]][all_query[0]]:
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

def get_excluded_bucket_idx(moco_model):
    moco_paths = moco_model.split(os.sep)
    model_configs = moco_paths[-2].split("_")
    excluded_bucket_idx = model_configs[model_configs.index('idx')+1]
    return int(excluded_bucket_idx)

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output
    
def make_model(train_mode, input_size=1024, output_size=1000):
    if train_mode in TRAIN_MODES_CATEGORY['mlp']:
        print(f"Using a mlp network with input size {input_size}")
        return MLP(input_size, 2048, output_size)
    elif train_mode in TRAIN_MODES_CATEGORY['linear']:
        print(f"Using a single linear layer")
        fc = torch.nn.Linear(input_size, output_size)
        fc.weight.data.normal_(mean=0.0, std=0.01)
        fc.bias.data.zero_()
        return fc
    elif train_mode in TRAIN_MODES_CATEGORY['cnn']:
        print(f"Using ResNet 50")
        pretrained = False
        selfsupervised = False
        if train_mode == 'cnn_imgnet':
            pretrained = True
        if train_mode == 'cnn_moco':
            selfsupervised = 'moco_v2'
        if train_mode == 'cnn_byol':
            selfsupervised = 'byol'
        if train_mode == "cnn_moco_yfcc_feb18_gpu_8_bucket_0":
            selfsupervised = "moco_v2_yfcc_feb18_bucket_0_gpu_8"
        model = training_utils.make_model(
            'resnet50',
            pretrained,
            selfsupervised,
            output_size=output_size
        )
        return model
    elif train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
        print(f"Using ResNet 50 (frozen feature extractor)")
        pretrained = False
        selfsupervised = False
        if train_mode == 'imgnet_linear':
            pretrained = True
        if train_mode == 'moco_v2_imgnet_linear':
            selfsupervised = 'moco_v2'
        if train_mode == 'byol_imgnet_linear':
            selfsupervised = 'byol'
        if train_mode == "moco_v2_yfcc_feb18_bucket_0_gpu_8_linear":
            selfsupervised = "moco_v2_yfcc_feb18_bucket_0_gpu_8"
        if train_mode == "moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_github":
            selfsupervised = "moco_v2_yfcc_feb18_bucket_0_gpu_8_github"
        model = training_utils.make_model(
            'resnet50',
            pretrained,
            selfsupervised,
            train_mode='freeze',
            output_size=output_size
        )
        # if train_mode in TRAIN_MODES_CATEGORY['cnn_mlp_feature']:
        #     model.fc = MLP(2048, 2048, output_size)
        # import pdb; pdb.set_trace()
        return model
    elif train_mode in TRAIN_MODES_CATEGORY['cnn_mlp_feature']:
        print(f"Using ResNet 50 (frozen feature extractor) with mlp")
        pretrained = False
        selfsupervised = False
        if train_mode == 'imgnet_mlp':
            pretrained = True
        if train_mode == 'moco_v2_imgnet_mlp':
            selfsupervised = 'moco_v2'
        if train_mode == 'byol_imgnet_mlp':
            selfsupervised = 'byol'
        if train_mode == "moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp":
            selfsupervised = "moco_v2_yfcc_feb18_bucket_0_gpu_8"
        model = training_utils.make_model(
            'resnet50',
            pretrained,
            selfsupervised,
            train_mode='freeze',
            output_size=output_size
        )
        if train_mode in TRAIN_MODES_CATEGORY['cnn_mlp_feature']:
            model.fc = MLP(2048, 2048, output_size)
        # import pdb; pdb.set_trace()
        return model
    else:
        raise NotImplementedError()

def make_optimizer(network, lr, weight_decay=1e-5):
    optimizer = torch.optim.SGD(
        list(filter(lambda x: x.requires_grad, network.parameters())),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9
    )
    return optimizer

def make_scheduler(optimizer, step_size=50):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=0.1
    )
    return scheduler

def get_loaders_from_loaders_dict(loaders_dict, train_mode):
    loaders = {}
    loaders['train'] = loaders_dict['train_set']
    if to_remove_random_crop(train_mode):
        loaders['train'] = remove_random_crop_from_loader(loaders['train'])
    loaders['test'] = loaders_dict['test_set']
    if 'val_set' in loaders_dict:
        loaders['val'] = loaders_dict['val_set']
    return loaders

def train(loaders, 
          train_mode, input_size, output_size,
          epochs=150, lr=0.1, weight_decay=1e-5, step_size=60):
    if train_mode in TRAIN_MODES_CATEGORY['nearest_mean']:
        network = {}
        # loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
        for batch, data in enumerate(loaders['train']):
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
        
        test_results = {set_name : test(loaders[set_name], network, train_mode) for set_name in loaders}
        return network, test_results
    else:
        network = make_model(train_mode, input_size, output_size).cuda()
        optimizer = make_optimizer(network, lr, weight_decay)
        scheduler = make_scheduler(optimizer, step_size=step_size)
        criterion = torch.nn.NLLLoss(reduction='mean')

        # TODO: Fix the loop
        # Try to 
        avg_results = {'train' : {'loss_per_epoch' : [], 'acc_per_epoch' : []},
                       'test' : {'loss_per_epoch' : [], 'acc_per_epoch' : []} }
        if 'val' in loaders:
            avg_results['val'] = {'loss_per_epoch' : [], 'acc_per_epoch' : []}
            model_selection_criterion = 'val'
        else:
            model_selection_criterion = 'test'

        best_result = {'best_acc' : 0, 'best_epoch' : None, 'best_network' : None}

        for epoch in range(0, epochs):
            print(f"Epoch {epoch}")
            for phase in loaders.keys():
                # import pdb; pdb.set_trace()
                if phase == 'train':
                    if train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature'] + TRAIN_MODES_CATEGORY['cnn_mlp_feature']:
                        print("Using Frozen Network in Eval Mode")
                        network.eval()
                    else:
                        network.train()
                else:
                    # import pdb; pdb.set_trace()
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
                    # pbar.set_postfix(loss=float(running_loss)/count, 
                    #                  acc=float(running_corrects)/count,
                    #                  epoch=epoch,
                    #                  phase=phase)
                    
                avg_loss = float(running_loss)/count
                avg_acc = float(running_corrects)/count
                avg_results[phase]['loss_per_epoch'].append(avg_loss)
                avg_results[phase]['acc_per_epoch'].append(avg_acc)
                if phase == 'train': 
                    scheduler.step()
                
                if phase == model_selection_criterion:
                    if avg_acc > best_result['best_acc']:
                        print(f"Best {model_selection_criterion} accuracy at epoch {epoch} being {avg_acc}")
                        best_result['best_epoch'] = epoch
                        best_result['best_acc'] = avg_acc
                        best_val_epoch_train_acc = avg_results['train']['acc_per_epoch'][-1]
                        print(f"Train accuracy at epoch {epoch} being {best_val_epoch_train_acc}")
                        best_result['best_network'] = copy.deepcopy(network.state_dict())
                
                print(f"Epoch {epoch}: Average {phase} Loss {avg_loss}, Accuracy {avg_acc:.2%}")
            print()
        print(
            f"Best Test Accuracy (for best {model_selection_criterion} model): {avg_results['test']['acc_per_epoch'][best_result['best_epoch']]:.2%}")
        print(f"Best Test Accuracy overall: {max(avg_results['test']['acc_per_epoch']):.2%}")
        network.load_state_dict(best_result['best_network'])
        test_acc = test(loaders['test'], network, train_mode, save_loc=None)
        print(f"Verify the best test accuracy for best {model_selection_criterion} is indeed {test_acc:.2%}")
        acc_result = {set_name : avg_results[set_name]['loss_per_epoch'][best_result['best_epoch']] 
                      for set_name in loaders.keys()}
        return network, acc_result, best_result, avg_results
        # {'train': best_val_epoch_train_acc, 'val': best_val_acc, 'test': test_acc}

def test(test_loader, network, train_mode, save_loc=None, class_names=None):
    # class_names should be sorted!!
    # If class_names != None, then return avg_acc, per_class_acc_dict
    
    if type(class_names) != type(None):
        assert sorted(class_names) == class_names
        idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}
        per_class_acc_dict = {idx: {'corrects': 0., 'counts' : 0.} for idx in idx_to_class.keys()}
    else:
        per_class_acc_dict = None

    if train_mode in TRAIN_MODES_CATEGORY['nearest_mean']:
        total_correct = 0.
        total_count = 0.
        pbar = test_loader
        for batch, data in enumerate(pbar):
            inputs, labels = data
            if train_mode in TRAIN_MODES_CATEGORY['feature_normalized']:
                inputs = inputs/torch.linalg.norm(inputs, dim=1, keepdim=True)
            for i in range(inputs.shape[0]):
                input_i = inputs[i]
                label_i = int(labels[i])
                pred_i = None
                max_score = None
                for class_i in network:
                    score = float(input_i.numpy().dot(network[class_i]))
                    if max_score == None or score > max_score:
                        max_score = score
                        pred_i = class_i
                if pred_i == label_i:
                    total_correct += 1
                if per_class_acc_dict != None:
                    per_class_acc_dict[label_i]['corrects'] += int(pred_i == label_i)
                    per_class_acc_dict[label_i]['counts'] += 1
                total_count += 1
        if per_class_acc_dict != None:
            per_class_acc_dict_copy = {}
            for idx in idx_to_class:
                per_class_acc_dict_copy[idx_to_class[idx]] = per_class_acc_dict[idx]
            return total_correct/total_count, per_class_acc_dict_copy
        else:
            return total_correct/total_count
    else:
        network = network.cuda().eval()
        if train_mode in TRAIN_MODES_CATEGORY['weight_normalized']:
            network.weight.data = network.weight.data/torch.linalg.norm(network.weight.data, dim=1, keepdim=True)
        running_corrects = 0.
        count = 0

        pbar = test_loader

        for batch, data in enumerate(pbar):
            inputs, labels = data
            if train_mode in TRAIN_MODES_CATEGORY['feature_normalized']:
                inputs = inputs/torch.linalg.norm(inputs, dim=1, keepdim=True)
            count += inputs.size(0)

            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.set_grad_enabled(False):
                outputs = network(inputs)
                _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)
            if per_class_acc_dict != None:
                for label_i, pred_i in zip(labels.data, preds):
                    per_class_acc_dict[int(label_i)]['corrects'] += int(pred_i == label_i)
                    per_class_acc_dict[int(label_i)]['counts'] += 1
            # pbar.set_postfix(acc=float(running_corrects)/count)

        avg_acc = float(running_corrects)/count
        print(f"Best Test Accuracy on test set: {avg_acc}")
        if save_loc:
            torch.save(network.state_dict(), save_loc)
        if per_class_acc_dict != None:
            per_class_acc_dict_copy = {}
            for idx in idx_to_class:
                per_class_acc_dict_copy[idx_to_class[idx]] = per_class_acc_dict[idx]
            return avg_acc, per_class_acc_dict_copy
        else:
            return avg_acc

def get_input_size(feature_name):
    if feature_name == 'clip_features':
        input_size = 1024
    # elif feature_name == 'moco_features':
    #     input_size = 2048
    elif feature_name == 'metadata':
        input_size = None
    return input_size

def avg_per_class_accuracy(per_class_accuracy_dict):
    total_count = 0.
    total_per_class_acc = 0.
    for class_name in per_class_accuracy_dict:
        total_per_class_acc += per_class_accuracy_dict[class_name]['corrects'] / per_class_accuracy_dict[class_name]['counts']
        total_count += 1.
    return total_per_class_acc/total_count

def only_positive_accuracy(per_class_accuracy_dict):
    total_count = 0.
    total_correct = 0.
    for class_name in per_class_accuracy_dict:
        if class_name != NEGATIVE_LABEL:
            total_count += per_class_accuracy_dict[class_name]['counts']
            total_correct += per_class_accuracy_dict[class_name]['corrects']
    return total_correct/total_count

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
                                         f"dataset_dict_{dataset_str(args.mode)}.pickle")
        loaders_dict_path = os.path.join(main_label_set_path,
                                         f"loaders_dict_{dataset_str(args.mode)}_{args.train_mode}.pickle")
        exclusive_loaders_dict_path = os.path.join(main_label_set_path,
                                                   f"exclusive_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_ex_{excluded_bucket_idx}.pickle")
        all_loaders_dict_path = os.path.join(main_label_set_path,
                                             f"all_loaders_dict_{dataset_str(args.mode)}_{args.train_mode}_ex_{excluded_bucket_idx}.pickle")
        if os.path.exists(dataset_dict_path):
            print(f"{dataset_dict_path} already exists.")
            dataset_dict = load_pickle(dataset_dict_path)
        else:
            dataset_dict = {}  # Saved the splitted dataset for each bucket

            for b_idx, sub_folder_path in enumerate(sub_folder_paths):
                dataset_dict_i_path = os.path.join(sub_folder_path,
                                                   f"dataset_dict_{b_idx}_{dataset_str(args.mode)}.pickle")
                if not os.path.exists(dataset_dict_i_path):
                    print(f"<<<<<<<<<<<First create split the dataset for bucket {b_idx}")
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
                                                   f"loaders_dict_{b_idx}_{dataset_str(args.mode)}_{args.train_mode}.pickle")
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
                                         f"results_dict_{dataset_str(args.mode)}_{args.train_mode}_ex_{excluded_bucket_idx}.pickle")
        
        if not os.path.exists(results_dict_path) or True: 
            all_query = sorted(list(dataset_dict[0].keys()))
            models_dict = {'models' : {}, 'b1_b2_accuracy_matrix' : {}, 'accuracy' : {},
                           'b1_b2_per_class_accuracy_dict' : {},
                           'only_positive_accuracy_test' : {},
                           'avg_per_class_accuracy_test' : {}}
            all_models_dict = {'models' : {}, 'accuracy_matrix' : {},
                               'per_class_accuracy_dict' : {},
                               'only_positive_accuracy_test' : {},
                               'avg_per_class_accuracy_test' : {}}
            results_dict = {'single' : models_dict, 'all' : all_models_dict}
            all_bucket = len(list(loaders_dict.keys()))
            if args.train_mode in TRAIN_MODES_CATEGORY['cnn'] + TRAIN_MODES_CATEGORY['cnn_linear_feature'] + TRAIN_MODES_CATEGORY['cnn_mlp_feature']:
                feature_name_list = ['metadata']
            elif args.train_mode in TRAIN_MODES_CATEGORY['linear'] + TRAIN_MODES_CATEGORY['nearest_mean'] + TRAIN_MODES_CATEGORY['mlp']:
                feature_name_list = ['clip_features']
            
            for feature_name in feature_name_list:
                single_accuracy_test = np.zeros((all_bucket, all_bucket))
                only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
                avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
                models_dict['models'][feature_name] = {}
                models_dict['accuracy'][feature_name] = {}
                b1_b2_per_class_accuracy_dict = {}
                input_size = get_input_size(feature_name)
                for b1 in range(all_bucket):
                    b1_b2_per_class_accuracy_dict[b1] = {}
                    loaders = get_loaders_from_loaders_dict(loaders_dict[b1][feature_name], args.train_mode)
                    single_model, single_accuracy_b1, _, _ = train(loaders, args.train_mode,
                                                                input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
                                                                lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay,
                                                                step_size=HYPER_DICT[args.train_mode].step)
                    models_dict['models'][feature_name][b1] = None
                    models_dict['accuracy'][feature_name][b1] = single_accuracy_b1
                    for b2 in range(all_bucket):
                        test_loader_b2 = loaders_dict[b2][feature_name]['test_set']
                        single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, args.train_mode, class_names=all_query)
                        b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
                        only_positive_accuracy_test[b1][b2] = only_positive_accuracy(per_class_accuracy_b1_b2)
                        avg_per_class_accuracy_test[b1][b2] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
                        print(
                            f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1][b2]:.4%} (pos only), {avg_per_class_accuracy_test[b1][b2]:.4%} (per class avg)")
                        single_accuracy_test[b1][b2] = single_accuracy_b1_b2
                models_dict['b1_b2_accuracy_matrix'][feature_name] = single_accuracy_test
                models_dict['b1_b2_per_class_accuracy_dict'][feature_name] = b1_b2_per_class_accuracy_dict
                models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy_test
                models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy_test
            

            for feature_name in feature_name_list:
                
                print(f"{feature_name}:")
                loaders = get_loaders_from_loaders_dict(all_loaders_dict[feature_name], args.train_mode)
                input_size = get_input_size(feature_name)
                all_model, all_accuracy, _, _ = train(loaders, args.train_mode, 
                                                input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
                                                lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay, step_size=HYPER_DICT[args.train_mode].step)
                print(all_accuracy)
                all_models_dict['accuracy_matrix'][feature_name] = all_accuracy
                all_models_dict['models'][feature_name] = all_model
                test_accuracy_all, per_class_accuracy_all = test(loaders['train'], all_model, args.train_mode, class_names=all_query)
                all_models_dict['per_class_accuracy_dict'][feature_name] = per_class_accuracy_all
                all_models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy(per_class_accuracy_all)
                all_models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy(per_class_accuracy_all)
                print(f"Baseline: {test_accuracy_all:.4%} (per sample), {all_models_dict['only_positive_accuracy_test'][feature_name]:.4%} (pos only), {all_models_dict['avg_per_class_accuracy_test'][feature_name]:.4%} (per class avg)")
            save_obj_as_pickle(results_dict_path, results_dict)
            print(f"Saved at {results_dict_path}")
        else:
            print(results_dict_path + " already exists")
            result_dict = load_pickle(results_dict_path)
            print(result_dict.keys())
            import pdb; pdb.set_trace()
        # assert os.path.exists(results_dict_path)
        # results_dict = load_pickle(results_dict_path)
        # all_query = sorted(list(dataset_dict[0].keys()))
        # all_models_dict = results_dict['all']
        # models_dict = results_dict['single']
        # all_bucket = len(list(loaders_dict.keys()))
        # if args.train_mode in TRAIN_MODES_CATEGORY['cnn'] + TRAIN_MODES_CATEGORY['cnn_linear_feature']:
        #     feature_name_list = ['metadata']
        # elif args.train_mode in TRAIN_MODES_CATEGORY['linear'] + TRAIN_MODES_CATEGORY['nearest_mean']:
        #     feature_name_list = ['clip_features']
        
        # if not 'b1_b2_per_class_accuracy_dict' in models_dict:
        #     models_dict['b1_b2_per_class_accuracy_dict'] = {}
        #     models_dict['only_positive_accuracy_test'] = {}
        #     models_dict['avg_per_class_accuracy_test'] = {}
        #     for feature_name in feature_name_list:
        #         only_positive_accuracy_test = np.zeros((all_bucket, all_bucket))
        #         avg_per_class_accuracy_test = np.zeros((all_bucket, all_bucket))
        #         b1_b2_per_class_accuracy_dict = {}
        #         for b1 in range(all_bucket):
        #             b1_b2_per_class_accuracy_dict[b1] = {}
        #             # train_loader = loaders_dict[b1][feature_name]['train_set']
        #             # if args.train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
        #             #     train_loader = remove_random_crop_from_loader(train_loader)
        #             # val_loader = loaders_dict[b1][feature_name]['val_set']
        #             test_loader = loaders_dict[b1][feature_name]['test_set']
        #             # single_model, single_accuracy_b1 = train(train_loader, val_loader, test_loader, args.train_mode,
        #             #                                          input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
        #             #                                          lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay, step_size=HYPER_DICT[args.train_mode].step)
        #             single_model = models_dict['models'][feature_name][b1]
        #             # models_dict['accuracy'][feature_name][b1] = single_accuracy_b1
        #             for b2 in range(all_bucket):
        #                 test_loader_b2 = loaders_dict[b2][feature_name]['test_set']
        #                 single_accuracy_b1_b2, per_class_accuracy_b1_b2 = test(test_loader_b2, single_model, args.train_mode, class_names=all_query)
        #                 b1_b2_per_class_accuracy_dict[b1][b2] = per_class_accuracy_b1_b2
        #                 only_positive_accuracy_test[b1][b2] = only_positive_accuracy(per_class_accuracy_b1_b2)
        #                 avg_per_class_accuracy_test[b1][b2] = avg_per_class_accuracy(per_class_accuracy_b1_b2)
        #                 print(
        #                     f"Train {b1}, test on {b2}: {single_accuracy_b1_b2:.4%} (per sample), {only_positive_accuracy_test[b1][b2]:.4%} (pos only), {avg_per_class_accuracy_test[b1][b2]:.4%} (per class avg)")
        #                 # single_accuracy_test[b1][b2] = single_accuracy_b1_b2
        #         # models_dict['b1_b2_accuracy_matrix'][feature_name] = single_accuracy_test
        #         models_dict['b1_b2_per_class_accuracy_dict'][feature_name] = b1_b2_per_class_accuracy_dict
        #         models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy_test
        #         models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy_test
        #     save_obj_as_pickle(results_dict_path, results_dict)
        #     print(f"Re-Saved at {results_dict_path}")
        
        # if not 'per_class_accuracy_dict' in all_models_dict or True:
        #     all_models_dict['per_class_accuracy_dict'] = {}
        #     all_models_dict['only_positive_accuracy_test'] = {}
        #     all_models_dict['avg_per_class_accuracy_test'] = {}
        #     for feature_name in feature_name_list:
        #         print(f"{feature_name}:")
        #         # train_loader = all_loaders_dict[feature_name]['train_set']
        #         # if args.train_mode in TRAIN_MODES_CATEGORY['cnn_linear_feature']:
        #         #     train_loader = remove_random_crop_from_loader(train_loader)
        #         # val_loader = all_loaders_dict[feature_name]['val_set']
        #         test_loader = all_loaders_dict[feature_name]['test_set']
        #         # input_size = get_input_size(feature_name)
        #         # all_model, all_accuracy = train(train_loader, val_loader, test_loader, args.train_mode, 
        #         #                                 input_size, len(all_query), epochs=HYPER_DICT[args.train_mode].epochs,
        #         #                                 lr=HYPER_DICT[args.train_mode].lr, weight_decay=HYPER_DICT[args.train_mode].weight_decay, step_size=HYPER_DICT[args.train_mode].step)
        #         all_model = all_models_dict['models'][feature_name]
        #         test_accuracy_all, per_class_accuracy_all = test(test_loader, all_model, args.train_mode, class_names=all_query)
        #         # print(all_accuracy)
        #         # all_models_dict['accuracy_matrix'][feature_name] = all_accuracy
        #         all_models_dict['per_class_accuracy_dict'][feature_name] = per_class_accuracy_all
        #         all_models_dict['only_positive_accuracy_test'][feature_name] = only_positive_accuracy(per_class_accuracy_all)
        #         all_models_dict['avg_per_class_accuracy_test'][feature_name] = avg_per_class_accuracy(per_class_accuracy_all)
        #         print(f"Baseline: {test_accuracy_all:.4%} (per sample), {all_models_dict['only_positive_accuracy_test'][feature_name]:.4%} (pos only), {all_models_dict['avg_per_class_accuracy_test'][feature_name]:.4%} (per class avg)")

        #     save_obj_as_pickle(results_dict_path, results_dict)
        #     print(f"Re-Saved at {results_dict_path}")
