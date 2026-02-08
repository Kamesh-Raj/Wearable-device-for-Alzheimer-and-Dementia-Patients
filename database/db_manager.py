"""
Database Manager - Hybrid Local Storage System
Combines SQLite for structured data with sqlite-vec for vector search
Ensures 100% privacy compliance with local-only storage.
Supports multi-patient and multi-caregiver architecture.
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import hashlib
import os

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    logging.warning("sqlite-vec not installed. Run: pip install sqlite-vec")


class DatabaseManager:
    """Manages SQLite database with integrated vector search using sqlite-vec"""
    
    def __init__(self, db_path: str = "data/neuro_assistive.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables and sqlite-vec extension"""
        try:
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Load sqlite-vec extension
            if SQLITE_VEC_AVAILABLE:
                self.connection.enable_load_extension(True)
                sqlite_vec.load(self.connection)
                self.connection.enable_load_extension(False)
            
            # Create tables
            self._create_tables()
            
            self.logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    # ========== Helper Methods (Reduce Code Duplication) ==========
    
    def _execute_query(self, query: str, params: tuple = (), fetch_one: bool = False, 
                      fetch_all: bool = False, commit: bool = False) -> Any:
        """
        Execute database query with consistent error handling
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: Return single row as dict
            fetch_all: Return all rows as list of dicts
            commit: Commit transaction after execution
            
        Returns:
            Query result or lastrowid
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            if commit:
                self.connection.commit()
                return cursor.lastrowid
            elif fetch_one:
                row = cursor.fetchone()
                return dict(row) if row else None
            elif fetch_all:
                return [dict(row) for row in cursor.fetchall()]
            else:
                return cursor
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            if commit:
                self.connection.rollback()
            raise
    
    def _dict_from_row(self, row) -> Optional[Dict[str, Any]]:
        """Convert sqlite3.Row to dictionary"""
        return dict(row) if row else None
    
    def _serialize_json(self, data: Any) -> Optional[str]:
        """Serialize data to JSON string"""
        return json.dumps(data) if data else None
    
    def _deserialize_json(self, json_str: str) -> Any:
        """Deserialize JSON string to Python object"""
        try:
            return json.loads(json_str) if json_str else None
        except json.JSONDecodeError:
            return None
    
    
    def _create_tables(self):
        """Create all required database tables"""
        cursor = self.connection.cursor()
        
        # Users table (Authentication)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('caregiver', 'patient', 'admin')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE, -- Link to users table if patient has login
                name TEXT NOT NULL,
                phone TEXT,
                email TEXT,
                photo_path TEXT,
                age INTEGER,
                condition TEXT,
                medical_history TEXT,
                emergency_contact TEXT,
                emergency_phone TEXT,
                details JSON, -- Additional medical history, notes
                config JSON, -- Preferences, thresholds
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # Caregivers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS caregivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL, -- Link to users table
                name TEXT NOT NULL,
                phone TEXT,
                email TEXT,
                details JSON, -- Contact info, etc.
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # Caregiver-Patient Relationship (Many-to-Many or One-to-Many)
        # Assuming one patient can have multiple caregivers, and one caregiver can care for multiple patients
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_caregivers (
                patient_id INTEGER,
                caregiver_id INTEGER,
                relationship_type TEXT, -- e.g., 'Primary', 'Family', 'Doctor'
                permissions JSON, -- What this caregiver can access/edit
                PRIMARY KEY (patient_id, caregiver_id),
                FOREIGN KEY(patient_id) REFERENCES patients(id),
                FOREIGN KEY(caregiver_id) REFERENCES caregivers(id)
            )
        """)
        
        # Known Persons table (metadata for vector embeddings) - Per Patient
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                relationship TEXT NOT NULL,
                photo_path TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP,
                recognition_count INTEGER DEFAULT 0,
                notes TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
        """)
        
        # Face embeddings using sqlite-vec
        if SQLITE_VEC_AVAILABLE:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS face_embeddings USING vec0(
                    embedding FLOAT[512],
                    person_id INTEGER,
                    patient_id INTEGER, -- To filter search by patient
                    image_path TEXT,
                    timestamp TEXT
                )
            """)
        
        # Reminders table (medications and appointments) - Per Patient
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                created_by_caregiver_id INTEGER, -- NULL if created by patient/system
                type TEXT NOT NULL CHECK (type IN ('medication', 'appointment', 'activity')),
                title TEXT NOT NULL,
                description TEXT,
                scheduled_time TIMESTAMP NOT NULL,
                repeat_pattern TEXT,  -- JSON: daily, weekly, etc.
                is_active BOOLEAN DEFAULT 1,
                last_triggered TIMESTAMP,
                trigger_count INTEGER DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(patient_id) REFERENCES patients(id),
                FOREIGN KEY(created_by_caregiver_id) REFERENCES caregivers(id)
            )
        """)
        
        # Patient Logs table (emotion and behavior tracking) - Per Patient
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                log_type TEXT NOT NULL CHECK (log_type IN ('emotion', 'gait', 'safety', 'interaction')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data JSON NOT NULL,  -- Flexible JSON storage for different log types
                severity_level INTEGER DEFAULT 0,  -- 0-10 scale
                location_lat REAL,
                location_lon REAL,
                notes TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
        """)
        
        # Safety Zones table (geofencing) - Per Patient
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                center_lat REAL NOT NULL,
                center_lon REAL NOT NULL,
                radius_meters INTEGER NOT NULL,
                zone_type TEXT DEFAULT 'safe' CHECK (zone_type IN ('safe', 'restricted', 'alert')),
                is_active BOOLEAN DEFAULT 1,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_caregivers BOOLEAN DEFAULT 1,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
        """)
        
        # Device Settings table (Global or Per Patient?) - Let's make it per patient if needed, but existing was global.
        # User requested "dashboard is for every patient... maintain their data without merging".
        # So settings should be per patient.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS device_settings (
                patient_id INTEGER,
                key TEXT,
                value TEXT NOT NULL,
                data_type TEXT DEFAULT 'string',
                updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                PRIMARY KEY (patient_id, key),
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
        """)
        
        # Caregiver Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS caregiver_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                alert_type TEXT NOT NULL,
                priority INTEGER DEFAULT 1,  -- 1=low, 2=medium, 3=high, 4=critical
                message TEXT NOT NULL,
                data JSON,
                is_acknowledged BOOLEAN DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged_date TIMESTAMP,
                acknowledged_by_caregiver_id INTEGER,
                FOREIGN KEY(patient_id) REFERENCES patients(id),
                FOREIGN KEY(acknowledged_by_caregiver_id) REFERENCES caregivers(id)
            )
        """)
        
        self.connection.commit()
        self.logger.info("Database tables created successfully (Multi-tenant support)")

    # --- User & Auth Management ---
    def create_user(self, username, password, role):
        """Create a new user (caregiver or patient)"""
        try:
            cursor = self.connection.cursor()
            # Simple hash for demo purposes - in production use bcrypt/argon2
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                           (username, password_hash, role))
            user_id = cursor.lastrowid
            self.connection.commit()
            return user_id
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return None

    def authenticate_user(self, username, password):
        """Authenticate user and return user info"""
        try:
            cursor = self.connection.cursor()
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute("SELECT id, role FROM users WHERE username = ? AND password_hash = ?", 
                           (username, password_hash))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            self.logger.error(f"Auth failed: {e}")
            return None

    def register_caregiver(self, user_id, name, phone=None, email=None, details=None):
        try:
            cursor = self.connection.cursor()
            cursor.execute("INSERT INTO caregivers (user_id, name, phone, email, details) VALUES (?, ?, ?, ?, ?)",
                           (user_id, name, phone, email, json.dumps(details or {})))
            cid = cursor.lastrowid
            self.connection.commit()
            return cid
        except Exception as e:
            self.logger.error(f"Failed to register caregiver: {e}")
            return None

    def register_patient(self, name, user_id=None, phone=None, email=None, details=None, config=None):
        try:
            cursor = self.connection.cursor()
            cursor.execute("INSERT INTO patients (user_id, name, phone, email, details, config) VALUES (?, ?, ?, ?, ?, ?)",
                           (user_id, name, phone, email, json.dumps(details or {}), json.dumps(config or {})))
            pid = cursor.lastrowid
            self.connection.commit()
            return pid
        except Exception as e:
            self.logger.error(f"Failed to register patient: {e}")
            return None

    def link_patient_caregiver(self, patient_id, caregiver_id, relationship_type="Primary"):
        try:
            cursor = self.connection.cursor()
            cursor.execute("INSERT OR REPLACE INTO patient_caregivers (patient_id, caregiver_id, relationship_type) VALUES (?, ?, ?)",
                           (patient_id, caregiver_id, relationship_type))
            self.connection.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to link patient-caregiver: {e}")
            return False

    def get_caregiver_patients(self, caregiver_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT p.*, pc.relationship_type 
                FROM patients p
                JOIN patient_caregivers pc ON p.id = pc.patient_id
                WHERE pc.caregiver_id = ?
            """, (caregiver_id,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get caregiver patients: {e}")
            return []

    # --- Known Persons Management ---
    def add_known_person(self, patient_id: int, name: str, relationship: str, photo_path: str = None, 
                        notes: str = None) -> int:
        """Add a new known person for a specific patient"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO known_persons (patient_id, name, relationship, photo_path, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (patient_id, name, relationship, photo_path, notes))
            
            person_id = cursor.lastrowid
            self.connection.commit()
            return person_id
        except Exception as e:
            self.logger.error(f"Failed to add known person: {e}")
            self.connection.rollback()
            return -1
    
    def add_face_embedding(self, person_id: int, patient_id: int, embedding: np.ndarray, 
                          image_path: str = None) -> bool:
        """Add face embedding for a known person using sqlite-vec"""
        if not SQLITE_VEC_AVAILABLE:
            self.logger.warning("sqlite-vec not available, skipping embedding storage")
            return False
        
        try:
            cursor = self.connection.cursor()
            embedding_array = np.array(embedding, dtype=np.float32)
            timestamp = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO face_embeddings (embedding, person_id, patient_id, image_path, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (embedding_array.tobytes(), person_id, patient_id, image_path, timestamp))
            
            self.connection.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to add face embedding: {e}")
            self.connection.rollback()
            return False
    
    def search_face_embedding(self, patient_id: int, embedding: np.ndarray, n_results: int = 5,
                             distance_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Search for similar face embeddings using sqlite-vec, filtered by patient"""
        if not SQLITE_VEC_AVAILABLE:
            return []
        
        try:
            cursor = self.connection.cursor()
            query_array = np.array(embedding, dtype=np.float32)
            
            # Note: sqlite-vec vector search with WHERE clause on other columns might need specific handling or post-filtering
            # However, standard SQL should work if the virtual table supports it.
            # Usually strict filtering with vector search is: SELECT ... WHERE patient_id = ? AND embedding MATCH ? 
            
            cursor.execute("""
                SELECT 
                    person_id,
                    image_path,
                    timestamp,
                    distance
                FROM face_embeddings
                WHERE embedding MATCH ? AND patient_id = ?
                ORDER BY distance
                LIMIT ?
            """, (query_array.tobytes(), patient_id, n_results))
            
            results = []
            for row in cursor.fetchall():
                person_id, image_path, timestamp, distance = row
                if distance <= distance_threshold:
                    person = self.get_known_person(person_id)
                    results.append({
                        'person_id': person_id,
                        'person_name': person['name'] if person else 'Unknown',
                        'relationship': person['relationship'] if person else 'Unknown',
                        'image_path': image_path,
                        'timestamp': timestamp,
                        'distance': distance,
                        'similarity': 1.0 - distance
                    })
            return results
        except Exception as e:
            self.logger.error(f"Failed to search face embedding: {e}")
            return []
    
    def get_known_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM known_persons WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Failed to get known person: {e}")
            return None
    
    def get_all_known_persons(self, patient_id: int) -> List[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM known_persons WHERE patient_id = ? ORDER BY name", (patient_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to get known persons: {e}")
            return []
    
    def update_person_recognition(self, person_id: int):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE known_persons 
                SET last_seen = CURRENT_TIMESTAMP, recognition_count = recognition_count + 1
                WHERE id = ?
            """, (person_id,))
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to update person recognition: {e}")

    # --- Reminders Management ---
    def add_reminder(self, patient_id: int, reminder_type: str, title: str, description: str,
                    scheduled_time: datetime, repeat_pattern: Dict[str, Any] = None, caregiver_id: int = None) -> int:
        try:
            cursor = self.connection.cursor()
            repeat_json = json.dumps(repeat_pattern) if repeat_pattern else None
            
            cursor.execute("""
                INSERT INTO reminders (patient_id, created_by_caregiver_id, type, title, description, scheduled_time, repeat_pattern)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, caregiver_id, reminder_type, title, description, scheduled_time, repeat_json))
            
            reminder_id = cursor.lastrowid
            self.connection.commit()
            return reminder_id
        except Exception as e:
            self.logger.error(f"Failed to add reminder: {e}")
            self.connection.rollback()
            return -1
    
    def get_due_reminders(self, patient_id: int, current_time: datetime = None) -> List[Dict[str, Any]]:
        if current_time is None:
            current_time = datetime.now()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM reminders 
                WHERE patient_id = ?
                AND is_active = 1 
                AND scheduled_time <= ? 
                AND (last_triggered IS NULL OR last_triggered < scheduled_time)
                ORDER BY scheduled_time
            """, (patient_id, current_time))
            
            rows = cursor.fetchall()
            reminders = []
            for row in rows:
                reminder = dict(row)
                if reminder['repeat_pattern']:
                    reminder['repeat_pattern'] = json.loads(reminder['repeat_pattern'])
                reminders.append(reminder)
            return reminders
        except Exception as e:
            self.logger.error(f"Failed to get due reminders: {e}")
            return []

    def get_patient_reminders(self, patient_id: int) -> List[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM reminders WHERE patient_id = ? ORDER BY scheduled_time", (patient_id,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            return []

    # --- Patient Logs Management ---
    def log_patient_data(self, patient_id: int, log_type: str, data: Dict[str, Any], 
                        severity_level: int = 0, location: Tuple[float, float] = None,
                        notes: str = None) -> int:
        try:
            cursor = self.connection.cursor()
            lat, lon = location if location else (None, None)
            data_json = json.dumps(data)
            
            cursor.execute("""
                INSERT INTO patient_logs (patient_id, log_type, data, severity_level, location_lat, location_lon, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, log_type, data_json, severity_level, lat, lon, notes))
            
            log_id = cursor.lastrowid
            self.connection.commit()
            return log_id
        except Exception as e:
            self.logger.error(f"Failed to log patient data: {e}")
            self.connection.rollback()
            return -1
    
    def get_patient_logs(self, patient_id: int, log_type: str = None, hours_back: int = 24,
                        severity_threshold: int = 0) -> List[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM patient_logs WHERE patient_id = ? AND timestamp >= ?"
            params = [patient_id, datetime.now() - timedelta(hours=hours_back)]
            
            if log_type:
                query += " AND log_type = ?"
                params.append(log_type)
            
            if severity_threshold > 0:
                query += " AND severity_level >= ?"
                params.append(severity_threshold)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            logs = []
            for row in rows:
                log_entry = dict(row)
                log_entry['data'] = json.loads(log_entry['data'])
                logs.append(log_entry)
            return logs
        except Exception as e:
            self.logger.error(f"Failed to get patient logs: {e}")
            return []
            
    # --- Safety Zones ---
    def add_safety_zone(self, patient_id: int, name: str, center_lat: float, center_lon: float,
                       radius_meters: int, zone_type: str = 'safe') -> int:
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO safety_zones (patient_id, name, center_lat, center_lon, radius_meters, zone_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (patient_id, name, center_lat, center_lon, radius_meters, zone_type))
            
            zone_id = cursor.lastrowid
            self.connection.commit()
            return zone_id
        except Exception as e:
            self.logger.error(f"Failed to add safety zone: {e}")
            return -1
            
    def get_active_safety_zones(self, patient_id: int) -> List[Dict[str, Any]]:
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM safety_zones WHERE patient_id = ? AND is_active = 1", (patient_id,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            return []

    def check_location_safety(self, patient_id: int, lat: float, lon: float) -> Dict[str, Any]:
        try:
            zones = self.get_active_safety_zones(patient_id)
            for zone in zones:
                # Simple distance calc - can use more precise haversine if needed (check prior version)
                # I'll reuse the prior _calculate_distance method if I had kept it, but I'll implement a simple one here
                # or better, reimplement the method I replaced.
                distance = self._calculate_distance(lat, lon, zone['center_lat'], zone['center_lon'])
                
                if distance <= zone['radius_meters']:
                    return {'is_safe': zone['zone_type'] == 'safe', 'zone': zone, 'distance': distance}
            
            return {'is_safe': False, 'zone': None, 'distance': None}
        except Exception as e:
            return {'is_safe': False, 'zone': None, 'distance': None}

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        a = (sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    # --- Alerts ---
    def create_alert(self, patient_id: int, alert_type: str, message: str, priority: int = 1,
                    data: Dict[str, Any] = None) -> int:
        try:
            cursor = self.connection.cursor()
            data_json = json.dumps(data) if data else None
            cursor.execute("""
                INSERT INTO caregiver_alerts (patient_id, alert_type, priority, message, data)
                VALUES (?, ?, ?, ?, ?)
            """, (patient_id, alert_type, priority, message, data_json))
            alert_id = cursor.lastrowid
            self.connection.commit()
            return alert_id
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
            return -1

    def get_unacknowledged_alerts(self, caregiver_id: int) -> List[Dict[str, Any]]:
        # Get alerts for all patients this caregiver cares for
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT ca.*, p.name as patient_name
                FROM caregiver_alerts ca
                JOIN patient_caregivers pc ON ca.patient_id = pc.patient_id
                JOIN patients p ON p.id = ca.patient_id
                WHERE pc.caregiver_id = ? AND ca.is_acknowledged = 0
                ORDER BY ca.priority DESC, ca.created_date DESC
            """, (caregiver_id,))
            
            alerts = []
            for row in cursor.fetchall():
                alert = dict(row)
                if alert['data']:
                    alert['data'] = json.loads(alert['data'])
                alerts.append(alert)
            return alerts
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            return []
    
    # ===== Patient Profile Management =====
    
    def get_patient_profile(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Get complete patient profile including photo and details"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, name, photo_path, age, condition, medical_history,
                       emergency_contact, emergency_phone, details, config
                FROM patients WHERE id = ?
            """, (patient_id,))
            
            row = cursor.fetchone()
            if row:
                profile = self._dict_from_row(row)
                # Deserialize JSON fields
                profile['details'] = self._deserialize_json(profile.get('details'))
                profile['config'] = self._deserialize_json(profile.get('config'))
                return profile
            return None
        except Exception as e:
            self.logger.error(f"Failed to get patient profile: {e}")
            return None
    
    def update_patient_profile(self, patient_id: int, profile_data: Dict[str, Any]) -> bool:
        """Update patient profile with new data"""
        try:
            cursor = self.connection.cursor()
            
            # Build update query dynamically based on provided fields
            allowed_fields = ['name', 'age', 'condition', 'medical_history', 
                            'emergency_contact', 'emergency_phone', 'details']
            
            update_fields = []
            values = []
            
            for field in allowed_fields:
                if field in profile_data:
                    update_fields.append(f"{field} = ?")
                    # Serialize JSON fields
                    if field == 'details' and isinstance(profile_data[field], (dict, list)):
                        values.append(self._serialize_json(profile_data[field]))
                    else:
                        values.append(profile_data[field])
            
            if not update_fields:
                return False
            
            values.append(patient_id)
            query = f"UPDATE patients SET {', '.join(update_fields)} WHERE id = ?"
            
            cursor.execute(query, values)
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to update patient profile: {e}")
            self.connection.rollback()
            return False
    
    def update_patient_photo(self, patient_id: int, photo_path: str) -> bool:
        """Update patient photo path"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("UPDATE patients SET photo_path = ? WHERE id = ?", 
                         (photo_path, patient_id))
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to update patient photo: {e}")
            self.connection.rollback()
            return False
    
    # ===== Safety Zone Management =====
    
    def add_safety_zone(self, patient_id: int, name: str, center_lat: float, 
                       center_lon: float, radius_meters: int, 
                       zone_type: str = 'safe') -> int:
        """Add a new safety zone for a patient"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO safety_zones (patient_id, name, center_lat, center_lon, 
                                         radius_meters, zone_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (patient_id, name, center_lat, center_lon, radius_meters, zone_type))
            zone_id = cursor.lastrowid
            self.connection.commit()
            return zone_id
        except Exception as e:
            self.logger.error(f"Failed to add safety zone: {e}")
            return -1
    
    def get_safety_zones(self, patient_id: int) -> List[Dict[str, Any]]:
        """Get all safety zones for a patient"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM safety_zones 
                WHERE patient_id = ? AND is_active = 1
                ORDER BY created_date DESC
            """, (patient_id,))
            return [self._dict_from_row(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get safety zones: {e}")
            return []
    
    def update_safety_zone(self, zone_id: int, updates: Dict[str, Any]) -> bool:
        """Update a safety zone"""
        try:
            cursor = self.connection.cursor()
            allowed_fields = ['name', 'center_lat', 'center_lon', 'radius_meters', 
                            'zone_type', 'is_active']
            
            update_fields = []
            values = []
            
            for field in allowed_fields:
                if field in updates:
                    update_fields.append(f"{field} = ?")
                    values.append(updates[field])
            
            if not update_fields:
                return False
            
            values.append(zone_id)
            query = f"UPDATE safety_zones SET {', '.join(update_fields)} WHERE id = ?"
            
            cursor.execute(query, values)
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to update safety zone: {e}")
            self.connection.rollback()
            return False
    
    def delete_safety_zone(self, zone_id: int) -> bool:
        """Delete (deactivate) a safety zone"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("UPDATE safety_zones SET is_active = 0 WHERE id = ?", (zone_id,))
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to delete safety zone: {e}")
            self.connection.rollback()
            return False
    
    def add_known_person(self, patient_id, name, relationship, photo_path, phone=None, email=None, notes=None, is_emergency_contact=False):
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO known_persons 
                (patient_id, name, relationship, photo_path, phone, email, notes, is_emergency_contact) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, name, relationship, photo_path, phone, email, notes, is_emergency_contact))
            self.connection.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to add known person: {e}")
            return False

    def get_known_persons(self, patient_id):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM known_persons WHERE patient_id = ?", (patient_id,))
            rows = cursor.fetchall()
            if not rows:
                return []
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to get known persons: {e}")
            return []

    def close(self):
        if self.connection:
            self.connection.close()
