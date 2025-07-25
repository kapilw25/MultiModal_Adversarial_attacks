#!/bin/bash

# Script to run all attacks sequentially on images from processed_images.json
# Modified for Multi-modal-Self-instruct project

echo "Starting attack execution sequence..."
echo "======================================"

# Activate virtual environment
source venv_MM/bin/activate

# Parse the processed_images.json file to get the list of images
echo "Loading image list from data/processed_images.json..."

# Initialize arrays
declare -a ALL_IMAGES=()
declare -a TASK_TYPES=("chart" "table" "road_map" "dashboard" "flowchart" "relation_graph" "planar_layout" "visual_puzzle")

# Function to extract images for a specific task from JSON
extract_images() {
    local task=$1
    local images=$(jq -r ".$task[]" data/processed_images.json)
    
    # Add each image to the ALL_IMAGES array with the proper path prefix
    for img in $images; do
        ALL_IMAGES+=("data/test_extracted/$img")
    done
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it to parse JSON."
    echo "You can install it with: sudo apt-get install jq"
    exit 1
fi

# Check if the JSON file exists
if [ ! -f "data/processed_images.json" ]; then
    echo "Error: data/processed_images.json not found!"
    exit 1
fi

# Extract images for each task
for task in "${TASK_TYPES[@]}"; do
    echo "Loading images for task: $task"
    extract_images "$task"
done

# Total number of images
TOTAL_IMAGES=${#ALL_IMAGES[@]}

echo "Running attacks on $TOTAL_IMAGES images across 8 tasks..."

# Loop through each image
for ((img_idx=0; img_idx<TOTAL_IMAGES; img_idx++)); do
    IMAGE_PATH="${ALL_IMAGES[$img_idx]}"
    IMG_NUM=$((img_idx+1))
    
    echo ""
    echo "Processing image $IMG_NUM/$TOTAL_IMAGES: $IMAGE_PATH"
    echo "------------------------------------------------------"
    
    echo "Running Transfer Attacks..."
    echo "--------------------------"

    # v2_pgd_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [1/17] Running PGD Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v2_pgd_attack.py --image_path $IMAGE_PATH --eps 0.02 --max_iter 50 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "PGD Attack completed for $IMAGE_PATH."

    # v3_fgsm_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [2/17] Running FGSM Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v3_fgsm_attack.py --image_path $IMAGE_PATH --eps 0.03 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "FGSM Attack completed for $IMAGE_PATH."

    # v4_cw_l2_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [3/17] Running CW L2 Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v4_cw_l2_attack.py --image_path $IMAGE_PATH --confidence 5 --max_iter 100 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "CW L2 Attack completed for $IMAGE_PATH."

    # v5_cw_l0_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [4/17] Running CW L0 Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v5_cw_l0_attack.py --image_path $IMAGE_PATH --max_iter 50 --confidence 10 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "CW L0 Attack completed for $IMAGE_PATH."

    # v6_cw_linf_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [5/17] Running CW Linf Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v6_cw_linf_attack.py --image_path $IMAGE_PATH --confidence 5 --max_iter 50 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "CW Linf Attack completed for $IMAGE_PATH."

    # v7_lbfgs_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [6/17] Running L-BFGS Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v7_lbfgs_attack.py --image_path $IMAGE_PATH --max_iter 100 --confidence 0.1 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "L-BFGS Attack completed for $IMAGE_PATH."

    # v8_jsma_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [7/17] Running JSMA Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v8_jsma_attack.py --image_path $IMAGE_PATH --max_iter 20 --theta 0.1 --max_pixel_change 10 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "JSMA Attack completed for $IMAGE_PATH."

    # v9_deepfool_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [8/17] Running DeepFool Attack on $IMAGE_PATH..."
    python attack_models/transfer_attacks/v9_deepfool_attack.py --image_path $IMAGE_PATH --max_iter 50 --overshoot 0.02 --targeted_regions --perceptual_constraint --ssim_threshold 0.85
    echo "DeepFool Attack completed for $IMAGE_PATH."

    echo "Running True Black-Box Attacks..."
    echo "-------------------------------"

    # v10_square_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [9/17] Running Square Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v10_square_attack.py --image_path $IMAGE_PATH --eps 0.15 --norm inf --max_iter 200 --p_init 0.3 --ssim_threshold 0.85
    echo "Square Attack completed for $IMAGE_PATH."

    # v11_hop_skip_jump_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [10/17] Running HopSkipJump Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v11_hop_skip_jump_attack.py --image_path $IMAGE_PATH --norm 2 --max_iter 50 --max_eval 1000 --ssim_threshold 0.85
    echo "HopSkipJump Attack completed for $IMAGE_PATH."

    # v12_pixel_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [11/17] Running Pixel Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v12_pixel_attack.py --image_path $IMAGE_PATH --th 10 --es 1 --max_iter 100 --ssim_threshold 0.85 --num_pixels 20
    echo "Pixel Attack completed for $IMAGE_PATH."

    # v13_simba_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [12/17] Running SimBA Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v13_simba_attack.py --image_path $IMAGE_PATH --epsilon 0.15 --max_iter 1000 --freq_dim 32 --ssim_threshold 0.85
    echo "SimBA Attack completed for $IMAGE_PATH."

    # v14_spatial_transformation_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [13/17] Running Spatial Transformation Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v14_spatial_transformation_attack.py --image_path $IMAGE_PATH --max_translation 3 --max_rotation 10 --max_scaling 0.1 --ssim_threshold 0.85
    echo "Spatial Transformation Attack completed for $IMAGE_PATH."

    # v15_query_efficient_bb_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [14/17] Running Query-Efficient Black-Box Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v15_query_efficient_bb_attack.py --image_path $IMAGE_PATH --num_basis 20 --sigma 0.015625 --max_iter 100 --epsilon 0.1 --ssim_threshold 0.85
    echo "Query-Efficient Black-Box Attack completed for $IMAGE_PATH."

    # v16_zoo_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [15/17] Running ZOO Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v16_zoo_attack.py --image_path $IMAGE_PATH --confidence 0.0 --learning_rate 1e-2 --max_iter 10 --binary_search_steps 1 --initial_const 1e-3 --nb_parallel 128 --variable_h 1e-4 --ssim_threshold 0.85
    echo "ZOO Attack completed for $IMAGE_PATH."

    # v17_boundary_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [16/17] Running Boundary Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v17_boundary_attack.py --image_path $IMAGE_PATH --delta 0.1 --epsilon 0.1 --max_iter 1000 --ssim_threshold 0.85
    echo "Boundary Attack completed for $IMAGE_PATH."

    # v18_geoda_attack
    echo "[$IMG_NUM/$TOTAL_IMAGES] [17/17] Running GeoDA Attack on $IMAGE_PATH..."
    python attack_models/true_black_box_attacks/v18_geoda_attack.py --image_path $IMAGE_PATH --norm 2 --sub_dim 10 --max_iter 1000 --ssim_threshold 0.85
    echo "GeoDA Attack completed for $IMAGE_PATH."

    echo "Completed all attacks for image $IMG_NUM/$TOTAL_IMAGES: $IMAGE_PATH"
    echo "======================================"
done

echo "All attacks completed successfully on all images!"
echo "Check the output directories for results."

# Deactivate virtual environment
deactivate
