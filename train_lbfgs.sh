# Best training loss model
# Test on dynamic_300_lbfgs
# Negative with test set
python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode linear
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode linear
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode linear
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode linear
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode linear

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_linear
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_linear
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_linear
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_linear
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_linear

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_linear
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_linear
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_linear
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_linear
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_linear

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_linear
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_linear
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_linear
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_linear
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_linear

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode mlp
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode mlp
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode mlp
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode mlp
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode mlp

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_mlp
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_mlp
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_mlp
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_mlp
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_imgnet_mlp

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_mlp
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_mlp
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_mlp
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_mlp
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode byol_imgnet_mlp

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_mlp
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_mlp
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_mlp
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_mlp
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode imgnet_mlp

python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp

CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_scratch

CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=5 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0

    #TODO
python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_imgnet
python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_byol
python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode default --train_mode cnn_moco

# Negative no test set
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode linear

CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_linear

CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_linear

CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_linear
CUDA_VISIBLE_DEVICES=1 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_linear

CUDA_VISIBLE_DEVICES=6 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=6 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=6 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=6 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear
CUDA_VISIBLE_DEVICES=6 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear

CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode mlp

CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_imgnet_mlp

CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_mlp
CUDA_VISIBLE_DEVICES=2 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode byol_imgnet_mlp

CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode imgnet_mlp

CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp

CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=3 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=4 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_scratch
CUDA_VISIBLE_DEVICES=4 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_scratch

CUDA_VISIBLE_DEVICES=4 python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=4 python train_lbfgs.py --seed 1 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=4 python train_lbfgs.py --seed 10 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=4 python train_lbfgs.py --seed 100 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0
CUDA_VISIBLE_DEVICES=4 python train_lbfgs.py --seed 1000 --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_moco_yfcc_feb18_gpu_8_bucket_0

    #TODO
python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_imgnet
python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_byol
python train_lbfgs.py --dataset_name dynamic_300_lbfgs --mode no_test_set --train_mode cnn_moco
