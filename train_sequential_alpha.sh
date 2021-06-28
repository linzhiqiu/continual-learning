# Best training loss model
# Test on dynamic_300

# Negative no test set
python train_sequential_alpha.py --alpha_value_mode exponential --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
python train_sequential_alpha.py --alpha_value_mode exponential --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
python train_sequential_alpha.py --alpha_value_mode exponential --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
python train_sequential_alpha.py --alpha_value_mode exponential --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned
python train_sequential_alpha.py --alpha_value_mode exponential --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_linear_tuned

python train_sequential_alpha.py --alpha_value_mode exponential --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_sequential_alpha.py --alpha_value_mode exponential --seed 1 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_sequential_alpha.py --alpha_value_mode exponential --seed 10 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_sequential_alpha.py --alpha_value_mode exponential --seed 100 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
python train_sequential_alpha.py --alpha_value_mode exponential --seed 1000 --dataset_name dynamic_300 --mode no_test_set --train_mode moco_v2_yfcc_feb18_bucket_0_gpu_8_mlp
