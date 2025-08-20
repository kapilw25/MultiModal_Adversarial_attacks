def safe_delete_file(file_path, reason):
    """
    Safely delete a file with user confirmation to prevent accidental deletion of valid files.
    
    Args:
        file_path (str): Path to the file to delete
        reason (str): Reason for deletion
        
    Returns:
        bool: True if file was deleted, False otherwise
    """
    import os
    print(f"⚠️  File validation failed: {file_path}")
    print(f"   Reason: {reason}")
    
    # Safe to auto-delete: empty files and obviously corrupted files
    safe_to_delete_reasons = [
        "File is empty",
        "Line count mismatch: has 0 entries",
        "Valid entry count mismatch: has 0 valid entries"
    ]
    
    # Check if this is safe to auto-delete
    if any(safe_reason in reason for safe_reason in safe_to_delete_reasons):
        try:
            os.remove(file_path)
            print(f"   ✅ Automatically deleted empty/corrupted file: {file_path}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to delete file: {e}")
            return False
    else:
        # For files with content but validation issues, be more careful
        print(f"   ⚠️  File has content but failed validation - NOT deleting automatically.")
        print(f"   💡 Please verify manually if this file should be regenerated.")
        return False

def validate_json_file(file_path, expected_count):
    """
    Validate that a JSON file exists, is not empty, and has the expected number of entries.
    
    Args:
        file_path (str): Path to the JSON file
        expected_count (int): Expected number of entries in the JSON file
        
    Returns:
        bool: True if file is valid, False if it should be deleted and regenerated
    """
    import os
    import json
    
    if not os.path.exists(file_path):
        return False
    
    try:
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            safe_delete_file(file_path, "File is empty")
            return False
        
        # Check if JSONL file is valid and has expected number of entries
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        print(f"🔍 Validating {file_path}: found {len(lines)} lines, expected {expected_count}")
        
        if len(lines) != expected_count:
            safe_delete_file(file_path, f"Line count mismatch: has {len(lines)} entries, expected {expected_count}")
            return False
        
        # Validate that each line is valid JSON with required fields
        valid_entries = 0
        required_fields = ['question_id', 'prompt', 'text', 'truth', 'type']
        
        for i, line in enumerate(lines):
            try:
                # Try to parse JSON for each line
                entry = json.loads(line)
                
                if not isinstance(entry, dict):
                    safe_delete_file(file_path, f"Line {i+1} is not a JSON object")
                    return False
                
                # Check for required fields
                missing_fields = [field for field in required_fields if field not in entry]
                if missing_fields:
                    safe_delete_file(file_path, f"Line {i+1} missing required fields: {missing_fields}")
                    return False
                
                valid_entries += 1
                
            except json.JSONDecodeError as e:
                safe_delete_file(file_path, f"Line {i+1} has invalid JSON: {e}")
                return False
        
        if valid_entries != expected_count:
            safe_delete_file(file_path, f"Valid entry count mismatch: has {valid_entries} valid entries, expected {expected_count}")
            return False
        
        print(f"✅ Valid file confirmed: {file_path} ({valid_entries} entries)")
        return True
        
    except Exception as e:
        safe_delete_file(file_path, f"Error during validation: {e}")
        return False

def select_attack(engine, task, random_count, auto_choice=None):
    """
    Interactive function to select attack type and determine output file and image path.
    Also checks if output files already exist to avoid redundant processing.
    Uses processed_images.json to determine which images to process.
    
    Args:
        engine (str): The model engine being used (e.g., 'gpt4o', 'Qwen25_VL_3B')
        task (str): The task type (e.g., 'chart')
        random_count (int): Number of samples
        auto_choice (int, optional): If provided, automatically select this option without prompting
        
    Returns:
        list: List of tuples (output_file, img_path, attack_name) or None if user wants to skip
    """
    import os
    import json
    
    # Load processed_images.json
    try:
        with open('data/processed_images.json', 'r') as f:
            processed_images = json.load(f)
        
        # Check if the task exists in the processed images
        if task not in processed_images:
            print(f"Error: Task '{task}' not found in processed_images.json")
            return None
            
        # Check if there are images for this task
        if not processed_images[task]:
            print(f"Error: No images found for task '{task}' in processed_images.json")
            return None
            
        print(f"Found {len(processed_images[task])} images for task '{task}' in processed_images.json")
    except FileNotFoundError:
        print("Error: data/processed_images.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in data/processed_images.json")
        return None
    
    # Define available attacks
    attacks = [
        {
            "name": "Original (No Attack)",
            "suffix": "",
            "img_dir": "data/test_extracted/"
        },
        {
            "name": "PGD Attack",
            "suffix": "_BB_pgd",
            "img_dir": "data/test_BB_pgd/"
        },
        {
            "name": "FGSM Attack",
            "suffix": "_BB_fgsm",
            "img_dir": "data/test_BB_fgsm/"
        },
        {
            "name": "CW-L2 Attack",
            "suffix": "_BB_cw_l2",
            "img_dir": "data/test_BB_cw_l2/"
        },
        {
            "name": "CW-L0 Attack",
            "suffix": "_BB_cw_l0",
            "img_dir": "data/test_BB_cw_l0/"
        },
        {
            "name": "CW-L∞ Attack",
            "suffix": "_BB_cw_linf",
            "img_dir": "data/test_BB_cw_linf/"
        },
        {
            "name": "L-BFGS Attack",
            "suffix": "_BB_lbfgs",
            "img_dir": "data/test_BB_lbfgs/"
        },
        {
            "name": "JSMA Attack",
            "suffix": "_BB_jsma",
            "img_dir": "data/test_BB_jsma/"
        },
        {
            "name": "DeepFool Attack",
            "suffix": "_BB_deepfool",
            "img_dir": "data/test_BB_deepfool/"
        },
        {
            "name": "Square Attack",
            "suffix": "_BB_square",
            "img_dir": "data/test_BB_square/"
        },
        {
            "name": "HopSkipJump Attack",
            "suffix": "_BB_hop_skip_jump",
            "img_dir": "data/test_BB_hop_skip_jump/"
        },
        {
            "name": "Pixel Attack",
            "suffix": "_BB_pixel",
            "img_dir": "data/test_BB_pixel/"
        },
        {
            "name": "SimBA Attack",
            "suffix": "_BB_simba",
            "img_dir": "data/test_BB_simba/"
        },
        {
            "name": "Spatial Transformation Attack",
            "suffix": "_BB_spatial",
            "img_dir": "data/test_BB_spatial/"
        },
        {
            "name": "Query-Efficient Black-box Attack",
            "suffix": "_BB_query_efficient_bb",
            "img_dir": "data/test_BB_query_efficient_bb/"
        },
        {
            "name": "ZOO Attack",
            "suffix": "_BB_zoo",
            "img_dir": "data/test_BB_zoo/"
        },
        {
            "name": "Boundary Attack",
            "suffix": "_BB_boundary",
            "img_dir": "data/test_BB_boundary/"
        },
        {
            "name": "GeoDA Attack",
            "suffix": "_BB_geoda",
            "img_dir": "data/test_BB_geoda/"
        }
    ]
    
    # Add option to run all attacks
    attacks.append({"name": "ALL ATTACKS", "suffix": "all", "img_dir": None})
    
    # If auto_choice is provided, use it directly
    if auto_choice is not None:
        choice = auto_choice
    else:
        # Display attack options
        print("\nSelect attack type:")
        for i, attack in enumerate(attacks):
            print(f"{i+1}. {attack['name']}")
        
        # Get user selection
        while True:
            try:
                choice = int(input("\nEnter your choice (number): "))
                if 1 <= choice <= len(attacks):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(attacks)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Process ALL ATTACKS option
    if choice == len(attacks):  # ALL ATTACKS option
        results = []
        for attack in attacks[:-1]:  # Exclude the ALL ATTACKS option itself
            output_file = f"results/models/{engine}/eval_{engine}_{task}_{random_count}{attack['suffix']}.json"
            
            # Check if file already exists and is valid
            if validate_json_file(output_file, random_count):
                print(f"Skipping {attack['name']} - Valid output file already exists: {output_file}")
                continue
                
            # Check if image directory exists
            if not os.path.exists(attack['img_dir']):
                print(f"Skipping {attack['name']} - Image directory not found: {attack['img_dir']}")
                continue
                
            # Verify that the processed images exist in this attack directory
            missing_images = []
            for img_path in processed_images[task]:
                full_path = os.path.join(attack['img_dir'], img_path)
                if not os.path.exists(full_path):
                    missing_images.append(img_path)
            
            if missing_images:
                print(f"Skipping {attack['name']} - {len(missing_images)} images not found in {attack['img_dir']}")
                continue
                
            results.append((output_file, attack['img_dir'], attack['name']))
        
        return results
    
    # Process single attack option
    else:
        attack = attacks[choice-1]
        output_file = f"results/models/{engine}/eval_{engine}_{task}_{random_count}{attack['suffix']}.json"
        
        # Check if file already exists and is valid
        if validate_json_file(output_file, random_count):
            print(f"Valid output file already exists: {output_file}")
            if auto_choice is None:  # Only ask if not in auto mode
                retry = input("Do you want to overwrite? (y/n): ").lower()
                if retry != 'y':
                    print("Skipping this attack.")
                    return None
            else:
                print("Auto mode: Skipping this attack.")
                return None
        
        # Check if image directory exists
        if not os.path.exists(attack['img_dir']):
            print(f"Error: Image directory not found: {attack['img_dir']}")
            return None
            
        # Verify that the processed images exist in this attack directory
        missing_images = []
        for img_path in processed_images[task]:
            full_path = os.path.join(attack['img_dir'], img_path)
            if not os.path.exists(full_path):
                missing_images.append(img_path)
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found in {attack['img_dir']}")
            print("Missing images:")
            for img in missing_images[:5]:  # Show first 5 missing images
                print(f"  - {img}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")
            
            if auto_choice is None:  # Only ask if not in auto mode
                proceed = input("Do you want to proceed anyway? (y/n): ").lower()
                if proceed != 'y':
                    print("Skipping this attack.")
                    return None
            else:
                print("Auto mode: Proceeding despite missing images.")
        
        # For single attack, return a list with one item for consistency
        return [(output_file, attack['img_dir'], attack['name'])]
