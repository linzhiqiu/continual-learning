# Best training loss model
# Test on dynamic_300

# Negative no test set
CUDA_VISIBLE_DEVICES=4 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode linear

CUDA_VISIBLE_DEVICES=4 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode linear_tuned
CUDA_VISIBLE_DEVICES=4 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode linear_tuned
CUDA_VISIBLE_DEVICES=4 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode linear_tuned
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode linear_tuned
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode linear_tuned

CUDA_VISIBLE_DEVICES=4 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=4 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear

CUDA_VISIBLE_DEVICES=4 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=4 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=4 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=4 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=4 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_linear_tuned

CUDA_VISIBLE_DEVICES=5 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=5 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=5 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear

CUDA_VISIBLE_DEVICES=5 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_linear_tuned

python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear
python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear
python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear

python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear_tuned
python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear_tuned
python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear_tuned
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear_tuned
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_linear_tuned

CUDA_VISIBLE_DEVICES=3 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=3 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=3 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=3 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=3 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear

CUDA_VISIBLE_DEVICES=3 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned

CUDA_VISIBLE_DEVICES=5 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned_2
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned_2
CUDA_VISIBLE_DEVICES=5 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned_2
CUDA_VISIBLE_DEVICES=5 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned_2
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned_2

CUDA_VISIBLE_DEVICES=5 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=5 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp

CUDA_VISIBLE_DEVICES=5 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode mlp_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode mlp_tuned

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_imgnet_mlp_tuned

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=5 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode byol_imgnet_mlp_tuned

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp_tuned
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode imgnet_mlp_tuned

python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp

python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp_tuned

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch

python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch_lower_lr
python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch_lower_lr
python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch_lower_lr
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch_lower_lr
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_scratch_lower_lr

python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0

CUDA_VISIBLE_DEVICES=6 python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
CUDA_VISIBLE_DEVICES=6 python train.py --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
CUDA_VISIBLE_DEVICES=6 python train.py --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr
CUDA_VISIBLE_DEVICES=6 python train.py --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0_lower_lr

    #TODO
python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_imgnet
python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_byol
python train.py --dataset_name dynamic_300 --mode no_test_set --train_mode cnn_moco
