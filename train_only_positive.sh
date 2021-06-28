# Best training loss model
# Test on dynamic_300_positive_only
# Negative no val set
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode linear_tuned

CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned

CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_linear_tuned

CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear

# python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear_tuned
# python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear_tuned
# python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear_tuned
# python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear_tuned
# python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_linear_tuned

CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned

CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode mlp


# python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
# python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
# python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
# python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
# python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned


# python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp_tuned
# python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp_tuned
# python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp_tuned
# python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp_tuned
# python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_imgnet_mlp

CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode byol_imgnet_mlp

CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp

CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=6 python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
# python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp_tuned
# python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp_tuned
# python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp_tuned
# python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp_tuned
# python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode imgnet_mlp_tuned


# python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
# python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
# python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
# python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
# python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned

python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch
python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch
python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch
python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch
python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch

python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch_lower_lr
python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch_lower_lr
python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch_lower_lr
python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch_lower_lr
python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_scratch_lower_lr

python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0

python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
python train_only_positive.py --seed 1 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
python train_only_positive.py --seed 10 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
python train_only_positive.py --seed 100 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
python train_only_positive.py --seed 1000 --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr

    #TODO
python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_imgnet
python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_byol
python train_only_positive.py --dataset_name dynamic_300_positive_only --mode no_test_set --train_mode cnn_moco
