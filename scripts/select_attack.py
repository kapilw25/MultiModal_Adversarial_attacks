def select_attack(engine, task, random_count):
    """
    Interactive function to select attack type and determine output file and image path.
    Also checks if output files already exist to avoid redundant processing.
    
    Args:
        engine (str): The model engine being used (e.g., 'gpt4o', 'Qwen25_VL_3B')
        task (str): The task type (e.g., 'chart')
        random_count (int): Number of samples
        
    Returns:
        list: List of tuples (output_file, img_path, attack_name) or None if user wants to skip
    """
    import os
    
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
        }
    ]
    
    # Add option to run all attacks
    attacks.append({"name": "ALL ATTACKS", "suffix": "all", "img_dir": None})
    
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
            output_file = f"results/{engine}/eval_{engine}_{task}_{random_count}{attack['suffix']}.json"
            
            # Check if file already exists
            if os.path.exists(output_file):
                print(f"Skipping {attack['name']} - Output file already exists: {output_file}")
                continue
                
            # Check if image directory exists
            if not os.path.exists(attack['img_dir']):
                print(f"Skipping {attack['name']} - Image directory not found: {attack['img_dir']}")
                continue
                
            results.append((output_file, attack['img_dir'], attack['name']))
        
        return results
    
    # Process single attack option
    else:
        attack = attacks[choice-1]
        output_file = f"results/{engine}/eval_{engine}_{task}_{random_count}{attack['suffix']}.json"
        
        # Check if file already exists
        if os.path.exists(output_file):
            print(f"Output file already exists: {output_file}")
            retry = input("Do you want to overwrite? (y/n): ").lower()
            if retry != 'y':
                print("Skipping this attack.")
                return None
        
        # For single attack, return a list with one item for consistency
        return [(output_file, attack['img_dir'], attack['name'])]
