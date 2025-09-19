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

def generate_output_path(engine, task, num_samples, attack):
    """
    Generate output file path following the new directory structure:
    results/inference/{engine}/[clean|adversarial/{attack_type}/{attack_name}/ssim_{threshold}]/{task}/eval_{engine}_{task}_{num_samples}.json

    Args:
        engine (str): Model engine name
        task (str): Task type
        num_samples (int): Total number of evaluation samples
        attack (dict): Attack configuration

    Returns:
        str: Complete output file path
    """
    if attack["attack_type"] == "clean":
        # Clean images: results/inference/{engine}/clean/{task}/
        output_path = f"results/inference/{engine}/clean/{task}/eval_{engine}_{task}_{num_samples}.json"
    else:
        # Adversarial images: results/inference/{engine}/adversarial/{attack_type}/{attack_name}/ssim_{threshold}/{task}/
        # Extract attack name and ssim from img_dir
        # img_dir format: "data/adversarial/whitebox/fgsm/ssim_085/"
        path_parts = attack["img_dir"].rstrip('/').split('/')
        attack_type = path_parts[2]  # whitebox or blackbox
        attack_name = path_parts[3]  # fgsm, pgd, etc.
        ssim_threshold = path_parts[4]  # ssim_085, etc.

        output_path = f"results/inference/{engine}/adversarial/{attack_type}/{attack_name}/{ssim_threshold}/{task}/eval_{engine}_{task}_{num_samples}.json"

    return output_path

def select_ssim_threshold(auto_choice=None):
    """
    Interactive function to select SSIM threshold for adversarial attacks.

    Args:
        auto_choice (int, optional): If provided, automatically select this option without prompting

    Returns:
        str or list: SSIM threshold directory name (e.g., 'ssim_085') or list of all thresholds for ALL option
    """
    ssim_options = [
        ("0.85", "ssim_085", "Standard threshold (0.85)"),
        ("0.90", "ssim_090", "High similarity (0.90)"),
        ("0.95", "ssim_095", "Very high similarity (0.95)"),
        ("ALL", ["ssim_085", "ssim_090", "ssim_095"], "ALL thresholds (0.85, 0.90, 0.95)")
    ]

    if auto_choice is not None:
        if 1 <= auto_choice <= len(ssim_options):
            _, ssim_dir, desc = ssim_options[auto_choice - 1]
            print(f"Selected SSIM threshold: {desc}")
            return ssim_dir
        else:
            # Default to ssim_085 if invalid choice
            print("Invalid SSIM choice, defaulting to 0.85")
            return "ssim_085"
    else:
        print("\nSelect SSIM threshold for adversarial attacks:")
        for i, (_, _, desc) in enumerate(ssim_options):
            print(f"  [{i+1}] {desc}")

        while True:
            try:
                choice = int(input(f"\nEnter your choice (1-{len(ssim_options)}): "))
                if 1 <= choice <= len(ssim_options):
                    _, ssim_dir, desc = ssim_options[choice - 1]
                    print(f"Selected SSIM threshold: {desc}")
                    return ssim_dir
                else:
                    print(f"Please enter a number between 1 and {len(ssim_options)}")
            except ValueError:
                print("Please enter a valid number")

def select_attack(engine, task, num_samples, auto_choice=None):
    """
    Interactive function to select attack type and determine output file and image path.
    Also checks if output files already exist to avoid redundant processing.
    Uses processed_images.json to determine which images to process.
    
    Args:
        engine (str): The model engine being used (e.g., 'gpt4o', 'Qwen25_VL_3B')
        task (str): The task type (e.g., 'chart')
        num_samples (int): Number of samples
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

    # Define available attacks with CORRECT whitebox/blackbox paths and SSIM integration
    attacks = []

    # Add clean attack (no SSIM dependency)
    attacks.append({
        "name": "Original (No Attack)",
        "img_dir": "data/clean/",
        "attack_type": "clean"
    })

    # WHITE-BOX ATTACKS (OPTIMIZED FOR 7.6GB GPU - 4 fast attacks)
    attacks.extend([
        {
            "name": "PGD",
            "img_dir": "data/adversarial/whitebox/pgd/",
            "attack_type": "whitebox"
        },
        {
            "name": "FGSM",
            "img_dir": "data/adversarial/whitebox/fgsm/",
            "attack_type": "whitebox"
        },
        {
            "name": "CW-Linf",
            "img_dir": "data/adversarial/whitebox/cw_linf/",
            "attack_type": "whitebox"
        },
        {
            "name": "DeepFool",
            "img_dir": "data/adversarial/whitebox/deepfool/",
            "attack_type": "whitebox"
        },
        # BLACK-BOX ATTACKS (OPTIMIZED FOR 7.6GB GPU - 5 fast attacks)
        {
            "name": "Square",
            "img_dir": "data/adversarial/blackbox/square/",
            "attack_type": "blackbox"
        },
        {
            "name": "SimBA",
            "img_dir": "data/adversarial/blackbox/simba/",
            "attack_type": "blackbox"
        },
        {
            "name": "Boundary",
            "img_dir": "data/adversarial/blackbox/boundary/",
            "attack_type": "blackbox"
        },
        {
            "name": "Pixel",
            "img_dir": "data/adversarial/blackbox/pixel/",
            "attack_type": "blackbox"
        },
        {
            "name": "Spatial",
            "img_dir": "data/adversarial/blackbox/spatial/",
            "attack_type": "blackbox"
        }
    ])

    # REMOVED SLOW ATTACKS (for GPU memory optimization):
    # - JSMA (593.75s avg) - Too computationally expensive
    # - CW-L0 (587.0s avg) - Too computationally expensive
    # - CW-L2 (259.5s avg) - Too computationally expensive
    # - L-BFGS (removed from attack_runner.py)
    # - HopSkipJump (914.75s avg) - Too computationally expensive
    # - Query-Efficient BB (not in current attack_runner.py)
    # - ZOO (3263.5s avg) - Too computationally expensive

    # Add option to run all attacks
    attacks.append({"name": "ALL ATTACKS", "img_dir": None})

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

    # Determine if SSIM selection is needed
    need_ssim = False
    if choice == len(attacks):  # ALL ATTACKS
        # Check if any attack is adversarial
        need_ssim = any(attack['attack_type'] != 'clean' for attack in attacks[:-1])
    else:
        # Check if selected attack is adversarial
        selected_attack = attacks[choice-1]
        need_ssim = selected_attack['attack_type'] != 'clean'

    # Get SSIM threshold selection only if needed
    if need_ssim:
        if auto_choice is not None:
            # Auto mode: use ALL SSIM thresholds for comprehensive evaluation
            ssim_selection = select_ssim_threshold(auto_choice=4)  # Use ALL thresholds
            print("Auto mode: Using ALL SSIM thresholds (0.85, 0.90, 0.95)")
        else:
            # Interactive mode: ask user for SSIM selection
            ssim_selection = select_ssim_threshold()

        # Handle ALL SSIM thresholds case
        if isinstance(ssim_selection, list):
            ssim_dirs = ssim_selection
            print(f"Will process {len(ssim_dirs)} SSIM thresholds: {', '.join([s.replace('ssim_0', '0.') for s in ssim_dirs])}")
        else:
            ssim_dirs = [ssim_selection]
    else:
        # Clean attack only - no SSIM needed
        ssim_dirs = ["ssim_085"]  # Default, won't be used
    
    # Process ALL ATTACKS option
    if choice == len(attacks):  # ALL ATTACKS option
        results = []
        for attack in attacks[:-1]:  # Exclude the ALL ATTACKS option itself
            if attack['attack_type'] == 'clean':
                # Clean attack - no SSIM dependency
                output_file = generate_output_path(engine, task, num_samples, attack)

                # Check if file already exists and is valid
                if validate_json_file(output_file, num_samples):
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
                    print(f"   Expected path structure: {attack['img_dir']}{task}/{{image}}")
                    continue

                results.append((output_file, attack['img_dir'], attack['name']))
            else:
                # Adversarial attack - process each SSIM threshold
                for ssim_dir in ssim_dirs:
                    # Create attack copy with SSIM directory
                    attack_with_ssim = attack.copy()
                    attack_with_ssim['img_dir'] = f"{attack['img_dir']}{ssim_dir}/"

                    output_file = generate_output_path(engine, task, num_samples, attack_with_ssim)

                    # Check if file already exists and is valid
                    if validate_json_file(output_file, num_samples):
                        print(f"Skipping {attack['name']} ({ssim_dir}) - Valid output file already exists: {output_file}")
                        continue

                    # Check if image directory exists
                    if not os.path.exists(attack_with_ssim['img_dir']):
                        print(f"Skipping {attack['name']} ({ssim_dir}) - Image directory not found: {attack_with_ssim['img_dir']}")
                        continue

                    # Verify that the processed images exist in this attack directory
                    missing_images = []
                    for img_path in processed_images[task]:
                        full_path = os.path.join(attack_with_ssim['img_dir'], img_path)
                        if not os.path.exists(full_path):
                            missing_images.append(img_path)

                    if missing_images:
                        print(f"Skipping {attack['name']} ({ssim_dir}) - {len(missing_images)} images not found in {attack_with_ssim['img_dir']}")
                        print(f"   Expected path structure: {attack_with_ssim['img_dir']}{task}/{{image}}")
                        continue

                    results.append((output_file, attack_with_ssim['img_dir'], attack['name']))

        return results
    
    # Process single attack option
    else:
        attack = attacks[choice-1]
        results = []

        if attack['attack_type'] == 'clean':
            # Clean attack - no SSIM dependency
            output_file = generate_output_path(engine, task, num_samples, attack)

            # Check if file already exists and is valid
            if validate_json_file(output_file, num_samples):
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
                print(f"Expected path structure: {attack['img_dir']}{task}/{{image}}")
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

            results.append((output_file, attack['img_dir'], attack['name']))
        else:
            # Adversarial attack - process each selected SSIM threshold
            for ssim_dir in ssim_dirs:
                # Create attack copy with SSIM directory
                attack_with_ssim = attack.copy()
                attack_with_ssim['img_dir'] = f"{attack['img_dir']}{ssim_dir}/"

                output_file = generate_output_path(engine, task, num_samples, attack_with_ssim)

                # Check if file already exists and is valid
                if validate_json_file(output_file, num_samples):
                    print(f"Valid output file already exists: {output_file}")
                    if auto_choice is None:  # Only ask if not in auto mode
                        retry = input(f"Do you want to overwrite {attack['name']} ({ssim_dir})? (y/n): ").lower()
                        if retry != 'y':
                            print(f"Skipping {attack['name']} ({ssim_dir}).")
                            continue
                    else:
                        print(f"Auto mode: Skipping {attack['name']} ({ssim_dir}).")
                        continue

                # Check if image directory exists
                if not os.path.exists(attack_with_ssim['img_dir']):
                    print(f"Error: Image directory not found: {attack_with_ssim['img_dir']}")
                    continue

                # Verify that the processed images exist in this attack directory
                missing_images = []
                for img_path in processed_images[task]:
                    full_path = os.path.join(attack_with_ssim['img_dir'], img_path)
                    if not os.path.exists(full_path):
                        missing_images.append(img_path)

                if missing_images:
                    print(f"Warning: {len(missing_images)} images not found in {attack_with_ssim['img_dir']}")
                    print(f"Expected path structure: {attack_with_ssim['img_dir']}{task}/{{image}}")
                    print("Missing images:")
                    for img in missing_images[:5]:  # Show first 5 missing images
                        print(f"  - {img}")
                    if len(missing_images) > 5:
                        print(f"  ... and {len(missing_images) - 5} more")

                    if auto_choice is None:  # Only ask if not in auto mode
                        proceed = input(f"Do you want to proceed with {attack['name']} ({ssim_dir}) anyway? (y/n): ").lower()
                        if proceed != 'y':
                            print(f"Skipping {attack['name']} ({ssim_dir}).")
                            continue
                    else:
                        print(f"Auto mode: Proceeding with {attack['name']} ({ssim_dir}) despite missing images.")

                results.append((output_file, attack_with_ssim['img_dir'], attack['name']))

        return results if results else None
