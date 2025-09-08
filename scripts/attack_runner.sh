#!/bin/bash

# Script to run all attacks sequentially on images from processed_images.json
# Modified for Multi-modal-Self-instruct project
# Enhanced to capture attack output parameters (SSIM, perturbations, etc.)

# Set up automatic logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/attack_logs.txt"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Clear previous log and start fresh
> "$LOG_FILE"

log_with_timestamp "Starting attack execution sequence with parameter capture..."
log_with_timestamp "============================================================"
log_with_timestamp "Log file: $LOG_FILE"

# Activate virtual environment
source venv_MM/bin/activate

# Global attack hyperparameters - CONSISTENT ACROSS ALL ATTACKS
SSIM_THRESHOLD="0.85"
COMMON_FLAGS="--targeted_regions --perceptual_constraint --ssim_threshold $SSIM_THRESHOLD"

# Initialize results directory and file
RESULTS_DIR="results"
ATTACK_RESULTS_FILE="$RESULTS_DIR/attack_parameters.json"
mkdir -p "$RESULTS_DIR"

# Create fresh attack_parameters.json file (with backup if exists)
if [ -f "$ATTACK_RESULTS_FILE" ]; then
    BACKUP_FILE="${ATTACK_RESULTS_FILE%.json}_backup_$(date +%Y%m%d_%H%M%S).json"
    cp "$ATTACK_RESULTS_FILE" "$BACKUP_FILE"
    log_with_timestamp "📁 Backed up existing attack results to: $BACKUP_FILE"
fi

# Initialize fresh JSON file with run metadata
EXECUTION_ID="run_$(date +%Y%m%d_%H%M%S)"
cat > "$ATTACK_RESULTS_FILE" << EOF
{
  "metadata": {
    "execution_id": "$EXECUTION_ID",
    "execution_date": "$(date -Iseconds)",
    "description": "Attack output parameters: SSIM, perturbations, execution metrics",
    "ssim_threshold": $SSIM_THRESHOLD,
    "total_images": 0,
    "completed_attacks": 0
  },
  "attacks": {}
}
EOF

log_with_timestamp "🆕 Created fresh attack_parameters.json with execution ID: $EXECUTION_ID"

# Function to extract parameter from log
extract_param() {
    local log_file=$1
    local pattern=$2
    local default=${3:-"0"}
    
    local value=$(grep "$pattern" "$log_file" | tail -1 | sed "s/.*$pattern[^:]*: \?\([0-9.-]*\).*/\1/")
    if [ -z "$value" ] || [ "$value" = "$log_file" ]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# Function to run attack and capture parameters
run_attack_with_capture() {
    local attack_name=$1
    local command=$2
    local image_path=$3
    local attack_num=$4
    local total_attacks=$5
    
    echo "[$attack_num/$total_attacks] Running $attack_name Attack on $image_path..."
    
    # Create temp log file
    local temp_log="/tmp/attack_${attack_name}_$$.log"
    local start_time=$(date +%s)
    
    # Run attack and capture output (show on terminal AND save to log)
    eval "$command" 2>&1 | tee "$temp_log"
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))
    
    # Extract parameters
    local ssim=$(extract_param "$temp_log" "SSIM" "0.0")
    local mean_pert=$(extract_param "$temp_log" "Mean perturbation" "0.0") 
    local max_pert=$(extract_param "$temp_log" "Max perturbation" "0.0")
    local l2_norm=$(extract_param "$temp_log" "L2 norm" "0.0")
    local l0_norm=$(extract_param "$temp_log" "L0 norm" "0")
    local queries=$(extract_param "$temp_log" "queries" "0")
    
    # Determine success
    local success="true"
    if [ $exit_code -ne 0 ] || grep -q -i "error\|failed\|exception" "$temp_log"; then
        success="false"
    fi
    
    # Get image info
    local task_type=$(echo "$image_path" | cut -d'/' -f3)
    local image_name=$(basename "$image_path")
    
    # Save to JSON using Python with environment variables (safer)
    export ATTACK_NAME="$attack_name"
    export IMAGE_PATH="$image_path"
    export IMAGE_NAME="$image_name"
    export TASK_TYPE="$task_type"
    export EXECUTION_TIME="$execution_time"
    export SUCCESS="$success"
    export SSIM_VAL="$ssim"
    export MEAN_PERT="$mean_pert"
    export MAX_PERT="$max_pert"
    export L2_NORM="$l2_norm"
    export L0_NORM="$l0_norm"
    export QUERIES="$queries"
    export RESULTS_FILE="$ATTACK_RESULTS_FILE"
    export EXECUTION_ID_VAL="$EXECUTION_ID"
    
    python3 << 'EOF'
import json
import os
from datetime import datetime

# Get values from environment variables
results_file = os.environ['RESULTS_FILE']
attack_name = os.environ['ATTACK_NAME']
image_path = os.environ['IMAGE_PATH']
image_name = os.environ['IMAGE_NAME']
task_type = os.environ['TASK_TYPE']
execution_time = int(os.environ['EXECUTION_TIME'])
success = os.environ['SUCCESS'] == 'true'
ssim_val = os.environ['SSIM_VAL']
mean_pert = os.environ['MEAN_PERT']
max_pert = os.environ['MAX_PERT']
l2_norm = os.environ['L2_NORM']
l0_norm = os.environ['L0_NORM']
queries = os.environ['QUERIES']
execution_id = os.environ['EXECUTION_ID_VAL']

# Helper function to safely convert values
def safe_float(val):
    if val in ['0', '0.0', 'N/A', '']:
        return None
    try:
        return float(val)
    except ValueError:
        return None

def safe_int(val):
    if val in ['0', '0.0', 'N/A', '']:
        return None
    try:
        return int(float(val))
    except ValueError:
        return None

# Load existing fresh JSON file
with open(results_file, 'r') as f:
    data = json.load(f)

# Initialize attack entry if not exists  
if attack_name not in data['attacks']:
    data['attacks'][attack_name] = {
        'attack_category': 'White-Box' if attack_name in ['PGD', 'FGSM', 'CW-L2', 'CW-L0', 'CW-Linf', 'L-BFGS', 'JSMA', 'DeepFool'] else 'Black-Box',
        'executions': []
    }

# Add execution data
execution_data = {
    'execution_id': execution_id,
    'image_path': image_path,
    'image_name': image_name, 
    'task_type': task_type,
    'execution_time_seconds': execution_time,
    'success': success,
    'timestamp': datetime.now().isoformat(),
    'parameters': {
        'ssim': safe_float(ssim_val),
        'mean_perturbation': safe_float(mean_pert),
        'max_perturbation': safe_float(max_pert),
        'l2_norm': safe_float(l2_norm),
        'l0_norm': safe_int(l0_norm),
        'total_queries': safe_int(queries)
    }
}

data['attacks'][attack_name]['executions'].append(execution_data)

# Update metadata
data['metadata']['last_updated'] = datetime.now().isoformat()
data['metadata']['completed_attacks'] += 1

# Write back to file
with open(results_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"[{attack_name}] Captured: SSIM={ssim_val}, Mean_Pert={mean_pert}, Success={success}")
EOF

    # Append to main log
    echo "=== $attack_name Attack ===" >> scripts/attack_logs.txt
    cat "$temp_log" >> scripts/attack_logs.txt
    echo "" >> scripts/attack_logs.txt
    
    # Clean up
    rm -f "$temp_log"
    
    echo "$attack_name Attack completed for $image_path."
}

# Parse the processed_images.json file to get the list of images
log_with_timestamp "Loading image list from data/processed_images.json..."

# Initialize arrays
declare -a ALL_IMAGES=()
declare -a TASK_TYPES=("chart" "table" "road_map" "dashboard" "flowchart" "relation_graph" "planar_layout" "visual_puzzle")

# Function to extract images for a specific task from JSON
extract_images() {
    local task=$1
    local images=$(jq -r ".$task[]" data/processed_images.json)
    
    # Add each image to the ALL_IMAGES array with the proper path prefix
    for img in $images; do
        ALL_IMAGES+=("data/clean/$img")
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

# Update metadata with total images count
python3 << EOF
import json
with open('$ATTACK_RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['metadata']['total_images'] = $TOTAL_IMAGES
with open('$ATTACK_RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
EOF

log_with_timestamp "Running attacks on $TOTAL_IMAGES images across 8 tasks..."

# Loop through each image
for ((img_idx=0; img_idx<TOTAL_IMAGES; img_idx++)); do
    IMAGE_PATH="${ALL_IMAGES[$img_idx]}"
    IMG_NUM=$((img_idx+1))
    
    echo ""
    echo "Processing image $IMG_NUM/$TOTAL_IMAGES: $IMAGE_PATH"
    echo "------------------------------------------------------"
    
    echo "Running White-box Attacks..."
    echo "---------------------------"

    # White-box attacks with parameter capture
    run_attack_with_capture "PGD" "python attack_models/white_box/v2_pgd_attack.py --image_path $IMAGE_PATH --eps 0.02 --max_iter 50 $COMMON_FLAGS" "$IMAGE_PATH" 1 17

    run_attack_with_capture "FGSM" "python attack_models/white_box/v3_fgsm_attack.py --image_path $IMAGE_PATH --eps 0.03 $COMMON_FLAGS" "$IMAGE_PATH" 2 17

    run_attack_with_capture "CW-L2" "python attack_models/white_box/v4_cw_l2_attack.py --image_path $IMAGE_PATH --confidence 5 --max_iter 100 $COMMON_FLAGS" "$IMAGE_PATH" 3 17

    run_attack_with_capture "CW-L0" "python attack_models/white_box/v5_cw_l0_attack.py --image_path $IMAGE_PATH --max_iter 50 --confidence 10 $COMMON_FLAGS" "$IMAGE_PATH" 4 17

    run_attack_with_capture "CW-Linf" "python attack_models/white_box/v6_cw_linf_attack.py --image_path $IMAGE_PATH --confidence 5 --max_iter 50 $COMMON_FLAGS" "$IMAGE_PATH" 5 17

    run_attack_with_capture "L-BFGS" "python attack_models/white_box/v7_lbfgs_attack.py --image_path $IMAGE_PATH --max_iter 100 --confidence 0.1 $COMMON_FLAGS" "$IMAGE_PATH" 6 17

    run_attack_with_capture "JSMA" "python attack_models/white_box/v8_jsma_attack.py --image_path $IMAGE_PATH --max_iter 20 --theta 0.1 --max_pixel_change 10 $COMMON_FLAGS" "$IMAGE_PATH" 7 17

    run_attack_with_capture "DeepFool" "python attack_models/white_box/v9_deepfool_attack.py --image_path $IMAGE_PATH --max_iter 50 --overshoot 0.02 $COMMON_FLAGS" "$IMAGE_PATH" 8 17

    echo "Running Black-box Attacks..."
    echo "----------------------------"

    # Black-box attacks with parameter capture
    run_attack_with_capture "Square" "python attack_models/black_box/v10_square_attack.py --image_path $IMAGE_PATH --eps 0.15 --norm inf --max_iter 200 --p_init 0.3 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 9 17

    run_attack_with_capture "HopSkipJump" "python attack_models/black_box/v11_hop_skip_jump_attack.py --image_path $IMAGE_PATH --norm 2 --max_iter 50 --max_eval 1000 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 10 17

    run_attack_with_capture "Pixel" "python attack_models/black_box/v12_pixel_attack.py --image_path $IMAGE_PATH --th 10 --es 1 --max_iter 100 --ssim_threshold $SSIM_THRESHOLD --num_pixels 20" "$IMAGE_PATH" 11 17

    run_attack_with_capture "SimBA" "python attack_models/black_box/v13_simba_attack.py --image_path $IMAGE_PATH --epsilon 0.15 --max_iter 1000 --freq_dim 32 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 12 17

    run_attack_with_capture "Spatial" "python attack_models/black_box/v14_spatial_transformation_attack.py --image_path $IMAGE_PATH --max_translation 3 --max_rotation 10 --max_scaling 0.1 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 13 17

    run_attack_with_capture "Query-Efficient-BB" "python attack_models/black_box/v15_query_efficient_bb_attack.py --image_path $IMAGE_PATH --num_basis 20 --sigma 0.015625 --max_iter 100 --epsilon 0.1 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 14 17

    run_attack_with_capture "ZOO" "python attack_models/black_box/v16_zoo_attack.py --image_path $IMAGE_PATH --confidence 0.0 --learning_rate 1e-2 --max_iter 10 --binary_search_steps 1 --initial_const 1e-3 --nb_parallel 128 --variable_h 1e-4 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 15 17

    run_attack_with_capture "Boundary" "python attack_models/black_box/v17_boundary_attack.py --image_path $IMAGE_PATH --delta 0.1 --epsilon 0.1 --max_iter 1000 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 16 17

    run_attack_with_capture "GeoDA" "python attack_models/black_box/v18_geoda_attack.py --image_path $IMAGE_PATH --norm 2 --sub_dim 10 --max_iter 1000 --ssim_threshold $SSIM_THRESHOLD" "$IMAGE_PATH" 17 17

    echo "Completed all attacks for image $IMG_NUM/$TOTAL_IMAGES: $IMAGE_PATH"
    echo "======================================"
done

log_with_timestamp "All attacks completed successfully on all images!"
log_with_timestamp "Check the output directories for results."

# Generate final summary report
echo ""
log_with_timestamp "Generating attack parameters summary report..."
log_with_timestamp "=============================================="

python3 << EOF
import json
import os
from statistics import mean, median
from collections import defaultdict

results_file = '$ATTACK_RESULTS_FILE'

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"\\nAttack Parameters Summary Report")
    print(f"Generated: {data['metadata']['execution_date']}")
    print(f"=" * 50)
    
    total_executions = 0
    successful_executions = 0
    attack_summary = {}
    
    for attack_name, attack_data in data['attacks'].items():
        executions = attack_data['executions']
        successful = [e for e in executions if e['success']]
        
        total_executions += len(executions)
        successful_executions += len(successful)
        
        if successful:
            ssim_values = [e['parameters']['ssim'] for e in successful if e['parameters']['ssim'] is not None]
            mean_pert_values = [e['parameters']['mean_perturbation'] for e in successful if e['parameters']['mean_perturbation'] is not None]
            exec_times = [e['execution_time_seconds'] for e in successful]
            
            attack_summary[attack_name] = {
                'category': attack_data['attack_category'],
                'success_rate': len(successful) / len(executions) * 100,
                'avg_ssim': mean(ssim_values) if ssim_values else 0,
                'avg_mean_pert': mean(mean_pert_values) if mean_pert_values else 0,
                'avg_exec_time': mean(exec_times)
            }
    
    print(f"\\nOverall Statistics:")
    print(f"- Total Executions: {total_executions}")
    print(f"- Successful Attacks: {successful_executions}")
    print(f"- Overall Success Rate: {successful_executions/total_executions*100:.1f}%")
    
    print(f"\\nAttack Performance Summary:")
    print(f"{'Attack':<20} {'Category':<10} {'Success%':<9} {'Avg SSIM':<9} {'Avg Pert':<9} {'Avg Time(s)':<12}")
    print(f"{'-'*20} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*12}")
    
    for attack, stats in sorted(attack_summary.items()):
        print(f"{attack:<20} {stats['category']:<10} {stats['success_rate']:<8.1f}% "
              f"{stats['avg_ssim']:<8.3f} {stats['avg_mean_pert']:<8.3f} {stats['avg_exec_time']:<11.1f}")
    
    # Category analysis
    whitebox_attacks = {k: v for k, v in attack_summary.items() if v['category'] == 'White-Box'}
    blackbox_attacks = {k: v for k, v in attack_summary.items() if v['category'] == 'Black-Box'}
    
    if whitebox_attacks:
        avg_whitebox_success = mean([a['success_rate'] for a in whitebox_attacks.values()])
        print(f"\\nWhite-Box Attacks Average Success Rate: {avg_whitebox_success:.1f}%")
    
    if blackbox_attacks:
        avg_blackbox_success = mean([a['success_rate'] for a in blackbox_attacks.values()])
        print(f"Black-Box Attacks Average Success Rate: {avg_blackbox_success:.1f}%")
    
    print(f"\\nDetailed results saved to: {results_file}")
    
else:
    print("No attack results found!")
EOF

# Deactivate virtual environment
deactivate

log_with_timestamp "============================================================"
log_with_timestamp "Attack execution sequence completed!"
log_with_timestamp "All logs have been saved to: $LOG_FILE"
log_with_timestamp "============================================================"
