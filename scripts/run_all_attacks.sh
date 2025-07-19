#!/bin/bash

# Script to run all attacks sequentially
# Created for Multi-modal-Self-instruct project

echo "Starting attack execution sequence..."
echo "======================================"

# Activate virtual environment
source venv_MM/bin/activate

# Set common image path
IMAGE_PATH="data/test_extracted/chart/20231114102825506748.png"

echo "Running Transfer Attacks..."
echo "--------------------------"

# v2_pgd_attack
echo "[1/17] Running PGD Attack..."
python attack_models/transfer_attacks/v2_pgd_attack.py --image_path $IMAGE_PATH --eps 0.02 --max_iter 50 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
echo "PGD Attack completed."

# v3_fgsm_attack
echo "[2/17] Running FGSM Attack..."
python attack_models/transfer_attacks/v3_fgsm_attack.py --image_path $IMAGE_PATH --eps 0.03 --targeted_regions --perceptual_constraint
echo "FGSM Attack completed."

# v4_cw_l2_attack
echo "[3/17] Running CW L2 Attack..."
python attack_models/transfer_attacks/v4_cw_l2_attack.py --image_path $IMAGE_PATH --confidence 5 --max_iter 100 --targeted_regions --perceptual_constraint
echo "CW L2 Attack completed."

# v5_cw_l0_attack
echo "[4/17] Running CW L0 Attack..."
python attack_models/transfer_attacks/v5_cw_l0_attack.py --image_path $IMAGE_PATH --max_iter 50 --confidence 10 --targeted_regions --perceptual_constraint
echo "CW L0 Attack completed."

# v6_cw_linf_attack
echo "[5/17] Running CW Linf Attack..."
python attack_models/transfer_attacks/v6_cw_linf_attack.py --image_path $IMAGE_PATH --confidence 5 --max_iter 50 --targeted_regions --perceptual_constraint
echo "CW Linf Attack completed."

# v7_lbfgs_attack
echo "[6/17] Running L-BFGS Attack..."
python attack_models/transfer_attacks/v7_lbfgs_attack.py --image_path $IMAGE_PATH --max_iter 100 --confidence 0.1 --targeted_regions --perceptual_constraint
echo "L-BFGS Attack completed."

# v8_jsma_attack
echo "[7/17] Running JSMA Attack..."
python attack_models/transfer_attacks/v8_jsma_attack.py --image_path $IMAGE_PATH --max_iter 20 --theta 0.1 --max_pixel_change 10 --targeted_regions --perceptual_constraint
echo "JSMA Attack completed."

# v9_deepfool_attack
echo "[8/17] Running DeepFool Attack..."
python attack_models/transfer_attacks/v9_deepfool_attack.py --image_path $IMAGE_PATH --max_iter 50 --overshoot 0.02 --targeted_regions --perceptual_constraint
echo "DeepFool Attack completed."

echo "Running True Black-Box Attacks..."
echo "-------------------------------"

# v10_square_attack
echo "[9/17] Running Square Attack..."
python attack_models/true_black_box_attacks/v10_square_attack.py --image_path $IMAGE_PATH --eps 0.15 --norm inf --max_iter 200 --p_init 0.3 --ssim_threshold 0.85
echo "Square Attack completed."

# v11_hop_skip_jump_attack
echo "[10/17] Running HopSkipJump Attack..."
python attack_models/true_black_box_attacks/v11_hop_skip_jump_attack.py --image_path $IMAGE_PATH --norm 2 --max_iter 50 --max_eval 1000 --ssim_threshold 0.85
echo "HopSkipJump Attack completed."

# v12_pixel_attack
echo "[11/17] Running Pixel Attack..."
python attack_models/true_black_box_attacks/v12_pixel_attack.py --image_path $IMAGE_PATH --th 10 --es 1 --max_iter 100 --ssim_threshold 0.85 --num_pixels 20
echo "Pixel Attack completed."

# v13_simba_attack
echo "[12/17] Running SimBA Attack..."
python attack_models/true_black_box_attacks/v13_simba_attack.py --image_path $IMAGE_PATH --epsilon 0.15 --max_iter 1000 --freq_dim 32 --ssim_threshold 0.85
echo "SimBA Attack completed."

# v14_spatial_transformation_attack
echo "[13/17] Running Spatial Transformation Attack..."
python attack_models/true_black_box_attacks/v14_spatial_transformation_attack.py --image_path $IMAGE_PATH --max_translation 3 --max_rotation 10 --max_scaling 0.1 --ssim_threshold 0.85
echo "Spatial Transformation Attack completed."

# v15_query_efficient_bb_attack
echo "[14/17] Running Query-Efficient Black-Box Attack..."
python attack_models/true_black_box_attacks/v15_query_efficient_bb_attack.py --image_path $IMAGE_PATH --num_basis 20 --sigma 0.015625 --max_iter 100 --epsilon 0.1 --ssim_threshold 0.85
echo "Query-Efficient Black-Box Attack completed."

# v16_zoo_attack
echo "[15/17] Running ZOO Attack..."
python attack_models/true_black_box_attacks/v16_zoo_attack.py --image_path $IMAGE_PATH --confidence 0.0 --learning_rate 1e-2 --max_iter 10 --binary_search_steps 1 --initial_const 1e-3 --nb_parallel 128 --variable_h 1e-4 --ssim_threshold 0.85
echo "ZOO Attack completed."

# v17_boundary_attack
echo "[16/17] Running Boundary Attack..."
python attack_models/true_black_box_attacks/v17_boundary_attack.py --image_path $IMAGE_PATH --delta 0.1 --epsilon 0.1 --max_iter 1000 --ssim_threshold 0.85
echo "Boundary Attack completed."

# v18_geoda_attack
echo "[17/17] Running GeoDA Attack..."
python attack_models/true_black_box_attacks/v18_geoda_attack.py --image_path $IMAGE_PATH --norm 2 --sub_dim 10 --max_iter 1000 --ssim_threshold 0.85
echo "GeoDA Attack completed."

echo "======================================"
echo "All attacks completed successfully!"
echo "Check the output directories for results."

# Deactivate virtual environment
deactivate
