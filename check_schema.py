"""
Check Database Schema
Shows the structure of all tables
"""

import sqlite3

db_path = "data/neuro_assistive.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'face_embeddings%'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print("Database Schema:\n")
    
    for table in tables:
        print(f"\n{'='*60}")
        print(f"Table: {table}")
        print('='*60)
        
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        for col in columns:
            col_id, name, type_, notnull, default, pk = col
            pk_str = " PRIMARY KEY" if pk else ""
            notnull_str = " NOT NULL" if notnull else ""
            default_str = f" DEFAULT {default}" if default else ""
            print(f"  {name:30} {type_:15} {pk_str}{notnull_str}{default_str}")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
