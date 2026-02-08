"""
Database Migration Script
Adds phone and email columns to existing caregivers table
"""

import sqlite3
import os

db_path = "data/neuro_assistive.db"

print("Starting database migration...")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(caregivers)")
    columns = [col[1] for col in cursor.fetchall()]
    
    print(f"Current caregivers table columns: {columns}")
    
    # Add phone column if it doesn't exist
    if 'phone' not in columns:
        cursor.execute("ALTER TABLE caregivers ADD COLUMN phone TEXT")
        print("+ Added 'phone' column to caregivers table")
    else:
        print("  'phone' column already exists")
    
    # Add email column if it doesn't exist
    if 'email' not in columns:
        cursor.execute("ALTER TABLE caregivers ADD COLUMN email TEXT")
        print("+ Added 'email' column to caregivers table")
    else:
        print("  'email' column already exists")
    
    conn.commit()
    conn.close()
    
    print("\n[SUCCESS] Migration complete!")
    print("   Existing user data has been preserved.")
    print("   New caregivers can now register with phone and email.")
    
except Exception as e:
    print(f"[ERROR] Migration failed: {e}")
