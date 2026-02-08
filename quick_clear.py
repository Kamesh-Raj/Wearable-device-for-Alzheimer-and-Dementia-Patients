"""
Quick Clear Database Script
"""
import sqlite3
import sys

db_path = "data/neuro_assistive.db"

try:
    conn = sqlite3.connect(db_path, timeout=5)
    cursor = conn.cursor()
    
    # Disable foreign keys
    cursor.execute("PRAGMA foreign_keys = OFF")
    
    # List of main tables to clear
    tables = [
        'users', 'caregivers', 'patients', 'patient_caregivers',
        'reminders', 'safety_zones', 'known_persons', 'patient_logs',
        'caregiver_alerts', 'device_settings', 'context_images'
    ]
    
    print("Clearing database tables...")
    for table in tables:
        try:
            cursor.execute(f"DELETE FROM {table}")
            print(f"  Cleared {table}")
        except Exception as e:
            print(f"  Skip {table}: {e}")
    
    # Reset sequences
    try:
        cursor.execute("DELETE FROM sqlite_sequence")
        print("  Reset auto-increment")
    except:
        pass
    
    cursor.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()
    
    print("\n[SUCCESS] Database cleared!")
    
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
