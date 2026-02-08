"""
Reset Database Script
This script deletes the old database and creates a new one with the updated schema.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager

def reset_database():
    db_path = "data/neuro_assistive.db"
    
    # Close any existing connections
    print("Resetting database...")
    
    # Remove old database if it exists
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"✓ Deleted old database: {db_path}")
        except Exception as e:
            print(f"✗ Could not delete database (it may be in use): {e}")
            print("  Please stop the Flask server and try again.")
            return False
    
    # Create new database with updated schema
    try:
        db = DatabaseManager(db_path)
        print("✓ Created new database with updated schema")
        print("✓ Tables created:")
        print("  - users (with username, password, role)")
        print("  - caregivers (with phone, email)")
        print("  - patients (with photo_path, age, condition, medical_history, emergency_contact)")
        print("  - safety_zones")
        print("  - reminders")
        print("  - known_persons")
        print("  - patient_logs")
        print("  - and more...")
        db.close()
        print("\n✅ Database reset complete! You can now register new users.")
        return True
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False

if __name__ == "__main__":
    reset_database()
