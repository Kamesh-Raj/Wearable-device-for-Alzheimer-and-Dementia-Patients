"""
Add columns to known_persons table
"""

import sqlite3

db_path = "data/neuro_assistive.db"

print("Adding columns to known_persons table...")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(known_persons)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Add phone column
    if 'phone' not in columns:
        cursor.execute("ALTER TABLE known_persons ADD COLUMN phone TEXT")
        print("+ Added 'phone' column")
    else:
        print("  'phone' already exists")
    
    # Add email column
    if 'email' not in columns:
        cursor.execute("ALTER TABLE known_persons ADD COLUMN email TEXT")
        print("+ Added 'email' column")
    else:
        print("  'email' already exists")
    
    # Add is_emergency_contact column
    if 'is_emergency_contact' not in columns:
        cursor.execute("ALTER TABLE known_persons ADD COLUMN is_emergency_contact BOOLEAN DEFAULT 0")
        print("+ Added 'is_emergency_contact' column")
    else:
        print("  'is_emergency_contact' already exists")
    
    conn.commit()
    conn.close()
    
    print("\n[SUCCESS] Migration complete!")
    
except Exception as e:
    print(f"[ERROR] {e}")
