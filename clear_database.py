"""
Clear All Database Tables
This script deletes all data from all tables while preserving the table structure.
"""

import sqlite3
import os

db_path = "data/neuro_assistive.db"

print("Clearing all data from database tables...")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\nFound {len(tables)} tables:")
    for table in tables:
        print(f"  - {table}")
    
    print("\nClearing data...")
    
    # Disable foreign key constraints temporarily
    cursor.execute("PRAGMA foreign_keys = OFF")
    
    # Delete all data from each table
    for table in tables:
        try:
            cursor.execute(f"DELETE FROM {table}")
            rows_deleted = cursor.rowcount
            print(f"  [OK] Cleared {rows_deleted} rows from '{table}'")
        except Exception as e:
            print(f"  [ERROR] Could not clear '{table}': {e}")
    
    # Re-enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Reset auto-increment counters
    cursor.execute("DELETE FROM sqlite_sequence")
    print("  [OK] Reset auto-increment counters")
    
    conn.commit()
    conn.close()
    
    print("\n[SUCCESS] All data cleared from database!")
    print("   Table structure preserved.")
    print("   You can now register new users from scratch.")
    
except Exception as e:
    print(f"\n[ERROR] Failed to clear database: {e}")
