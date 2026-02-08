"""
Database Migration Script - Add Patient Contact Info
Adds phone and email columns to existing patients table
"""

import sqlite3
import os

db_path = "data/neuro_assistive.db"

print("Starting database migration for patients table...")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(patients)")
    columns = [col[1] for col in cursor.fetchall()]
    
    print(f"Current patients table columns: {columns}")
    
    # Add phone column if it doesn't exist
    if 'phone' not in columns:
        cursor.execute("ALTER TABLE patients ADD COLUMN phone TEXT")
        print("+ Added 'phone' column to patients table")
    else:
        print("  'phone' column already exists")
    
    # Add email column if it doesn't exist
    if 'email' not in columns:
        cursor.execute("ALTER TABLE patients ADD COLUMN email TEXT")
        print("+ Added 'email' column to patients table")
    else:
        print("  'email' column already exists")
    
    conn.commit()
    conn.close()
    
    print("\n[SUCCESS] Migration complete!")
    print("   Patients can now register with phone and email for emergency alerts.")
    
except Exception as e:
    print(f"[ERROR] Migration failed: {e}")
