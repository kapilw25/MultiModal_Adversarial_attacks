#!/usr/bin/env python3
"""
Test database creation process to debug table creation failure
"""

import os
import sqlite3
import sys
sys.path.append('/lambda/nfs/DiskUsEast1/MultiModal_Adversarial_attacks')

def test_database_creation():
    """Test the exact sequence that attack_runner.py follows"""

    db_path = "results/centralized_pipeline.db"

    # Step 1: Clean slate
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🗑️  Removed existing database")

    # Step 2: Create CentralizedDB instance (what attack_runner.py does)
    print("🔧 Creating CentralizedDB instance...")
    from scripts.utils.centralized_database import CentralizedDB

    db = CentralizedDB()
    print("✅ CentralizedDB instance created")

    # Step 3: Verify table exists immediately after creation
    print("🔍 Checking tables immediately after CentralizedDB creation...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"📊 Tables found: {tables}")

    if 'attack_executions' in tables:
        print("✅ attack_executions table exists after CentralizedDB creation")
    else:
        print("❌ attack_executions table MISSING after CentralizedDB creation!")
        return False

    # Step 4: Test a database operation (what attack_runner.py does)
    print("🧪 Testing database insert operation...")

    try:
        test_data = {
            'image_path': 'data/clean/chart/test.png',
            'attack_type': 'square',
            'attack_category': 'blackbox',
            'task_type': 'chart',
            'epsilon_level': 'minimal',
            'epsilon_target': 4/255,
            'epsilon_achieved': 0.016,
            'adversarial_image_path': 'data/adversarial/blackbox/square/eps_0016/chart/test.png',
            'success': True,
            'execution_time': 10,
            'mean_perturbation': 0.5,
            'max_perturbation': 4.0,
            'l2_norm': 5.0,
            'l0_norm': 1000,
            'queries_used': 100,
            'trial_number': 1
        }

        db.insert_attack_result(test_data)
        print("✅ Database insert successful")

        # Verify data was inserted
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attack_executions")
        count = cursor.fetchone()[0]
        conn.close()

        print(f"📊 Records in attack_executions: {count}")

        if count > 0:
            print("🎉 SUCCESS: Database creation and insertion working correctly!")
            return True
        else:
            print("❌ FAILURE: No records inserted")
            return False

    except Exception as e:
        print(f"❌ Database operation failed: {e}")

        # Check if table still exists after the error
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            print(f"📊 Tables after error: {tables}")
        except Exception as e2:
            print(f"❌ Can't even check tables: {e2}")

        return False

if __name__ == "__main__":
    success = test_database_creation()
    sys.exit(0 if success else 1)