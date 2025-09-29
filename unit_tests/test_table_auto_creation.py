#!/usr/bin/env python3
"""
Test that database tables are automatically created/recreated when missing
"""

import os
import sqlite3
import sys
sys.path.append('/lambda/nfs/DiskUsEast1/MultiModal_Adversarial_attacks')

def test_automatic_table_creation():
    """Test that tables are automatically recreated when missing"""

    db_path = "results/centralized_pipeline.db"

    # Step 1: Create a database with tables
    print("🔧 Creating initial database...")
    from scripts.utils.centralized_database import CentralizedDB

    db = CentralizedDB()
    print("✅ Initial database created")

    # Step 2: Verify tables exist
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"📊 Initial tables: {tables}")
    assert 'attack_executions' in tables, "attack_executions table should exist initially"

    # Step 3: Manually delete the attack_executions table (simulating corruption/manual deletion)
    print("🗑️  Simulating table deletion...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE attack_executions")
    conn.commit()
    conn.close()

    # Verify table is gone
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"📊 Tables after deletion: {tables}")
    assert 'attack_executions' not in tables, "attack_executions table should be deleted"

    # Step 4: Try to insert data using the same db instance - should auto-recreate table
    print("🧪 Testing automatic table recreation...")

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

    try:
        db.insert_attack_result(test_data)
        print("✅ Insert successful - table was auto-recreated!")

        # Verify table exists again
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Verify data was inserted
        cursor.execute("SELECT COUNT(*) FROM attack_executions")
        count = cursor.fetchone()[0]
        conn.close()

        print(f"📊 Tables after auto-recreation: {tables}")
        print(f"📊 Records in attack_executions: {count}")

        if 'attack_executions' in tables and count > 0:
            print("🎉 SUCCESS: Automatic table creation works!")
            return True
        else:
            print("❌ FAILURE: Table not recreated or no data")
            return False

    except Exception as e:
        print(f"❌ Auto-recreation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_automatic_table_creation()
    sys.exit(0 if success else 1)