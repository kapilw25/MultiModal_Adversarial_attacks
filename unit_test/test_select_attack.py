#!/usr/bin/env python3
"""
Test script for the select_attack function.
This script uses mock inputs to test the function without requiring user interaction.
"""

import sys
import os
import unittest
from unittest.mock import patch
import io

# Add the scripts directory to the path so we can import select_attack
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from select_attack import select_attack

class TestSelectAttack(unittest.TestCase):
    """Test cases for the select_attack function"""
    
    @patch('builtins.input', side_effect=['1', 'y'])  # Mock user selecting option 1 (Original) and confirming overwrite
    @patch('os.path.exists', return_value=True)  # Mock file already exists
    def test_select_original_attack_overwrite(self, mock_exists, mock_input):
        """Test selecting the Original (No Attack) option with file overwrite"""
        result = select_attack("test_engine", "chart", 5)
        
        # Check that we got a list with one tuple
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        # Check the tuple contents
        output_file, img_dir, attack_name = result[0]
        self.assertEqual(output_file, "results/test_engine/eval_test_engine_chart_5.json")
        self.assertEqual(img_dir, "data/test_extracted/")
        self.assertEqual(attack_name, "Original (No Attack)")
    
    @patch('builtins.input', side_effect=['2'])  # Mock user selecting option 2 (PGD)
    @patch('os.path.exists', return_value=False)  # Mock file doesn't exist
    def test_select_pgd_attack_new_file(self, mock_exists, mock_input):
        """Test selecting the PGD Attack option with new file"""
        result = select_attack("test_engine", "chart", 5)
        
        # Check that we got a list with one tuple
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        # Check the tuple contents
        output_file, img_dir, attack_name = result[0]
        self.assertEqual(output_file, "results/test_engine/eval_test_engine_chart_5_adv.json")
        self.assertEqual(img_dir, "data/test_extracted_adv/")
        self.assertEqual(attack_name, "PGD Attack")
    
    @patch('builtins.input', side_effect=['7'])  # Mock user selecting option 7 (ALL ATTACKS)
    def test_select_all_attacks(self, mock_input):
        """Test selecting the ALL ATTACKS option"""
        # We need to patch os.path.exists to return False for all files
        # and os.path.exists to return True for all directories
        with patch('os.path.exists', side_effect=lambda path: not path.endswith('.json')):
            result = select_attack("test_engine", "chart", 5)
            
            # Check that we got a list with multiple tuples (one for each attack type)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 6)  # 6 attack types
            
            # Check the first tuple (Original)
            output_file, img_dir, attack_name = result[0]
            self.assertEqual(output_file, "results/test_engine/eval_test_engine_chart_5.json")
            self.assertEqual(img_dir, "data/test_extracted/")
            self.assertEqual(attack_name, "Original (No Attack)")
            
            # Check the last tuple (CW-L∞)
            output_file, img_dir, attack_name = result[5]
            self.assertEqual(output_file, "results/test_engine/eval_test_engine_chart_5_adv_cw_linf.json")
            self.assertEqual(img_dir, "data/test_extracted_adv_cw_linf/")
            self.assertEqual(attack_name, "CW-L∞ Attack")

def manual_test():
    """Run a manual test with real user input"""
    print("Testing select_attack function with manual input...")
    
    # Test parameters
    engine = "test_engine"
    task = "chart"
    random_count = 5
    
    # Print the expected behavior
    print("\nExpected behavior:")
    print("1. Function will display a list of attack options")
    print("2. User selects an attack type by number")
    print("3. Function returns a list of tuples with (output_file, img_dir, attack_name)")
    print("4. If 'ALL ATTACKS' is selected, it returns multiple tuples")
    print("5. If a file already exists, it asks for confirmation to overwrite")
    
    print("\nActual behavior:")
    print("Please follow the prompts to test the function.")
    
    # Call the function
    result = select_attack(engine, task, random_count)
    
    # Display the result
    print("\nFunction returned:")
    if result is None:
        print("None (skipped)")
    else:
        for i, (output_file, img_dir, attack_name) in enumerate(result):
            print(f"Result {i+1}:")
            print(f"  Output file: {output_file}")
            print(f"  Image directory: {img_dir}")
            print(f"  Attack name: {attack_name}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    # Check if we should run automated tests or manual test
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        manual_test()
    else:
        # Run the automated tests
        unittest.main()
