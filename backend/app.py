from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import os
import sys
import json
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import logging

# Add project root to path to import database.db_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_manager import DatabaseManager

app = Flask(__name__)
app.secret_key = 'super_secret_key_change_me'  # Use environment variable in production
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Database
db = DatabaseManager()

# --- Helpers ---
def login_required(role=None):
    def wrapper(f):
        from functools import wraps
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if role and session.get('role') != role and session.get('role') != 'admin':
                return "Unauthorized", 403
            return f(*args, **kwargs)
        return decorated_function
    return wrapper

def get_current_user_id():
    return session.get('user_id')

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = db.authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

def validate_username(username):
    """Validate username: only letters, numbers, and underscores"""
    import re
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores (_). No spaces allowed."
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    return True, ""

def validate_password(password):
    """Validate password: must contain uppercase, lowercase, number, and symbol"""
    import re
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number."
    if not re.search(r'[@$!%*?&]', password):
        return False, "Password must contain at least one special character (@$!%*?&)."
    return True, ""

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        name = request.form['name']
        
        # Validate username
        valid, error_msg = validate_username(username)
        if not valid:
            flash(error_msg)
            return render_template('register.html')
        
        # Validate password
        valid, error_msg = validate_password(password)
        if not valid:
            flash(error_msg)
            return render_template('register.html')
        
        user_id = db.create_user(username, password, role)
        if user_id:
            if role == 'caregiver':
                phone = request.form.get('phone')
                email = request.form.get('email')
                db.register_caregiver(user_id, name, phone, email)
            elif role == 'patient':
                phone = request.form.get('patient_phone')
                email = request.form.get('patient_email')
                db.register_patient(name, user_id=user_id, phone=phone, email=email)
            
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Username might already be taken.')
    return render_template('register.html')

@app.route('/dashboard')
@login_required()
def dashboard():
    user_id = session['user_id']
    role = session['role']
    
    if role == 'caregiver':
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM caregivers WHERE user_id = ?", (user_id,))
        caregiver = cursor.fetchone()
        
        if not caregiver:
            return "Caregiver profile not found", 404
            
        caregiver_id = caregiver['id']
        # This will be passed to template
        patients = db.get_caregiver_patients(caregiver_id)
        
        # We need all alerts for all patients of this caregiver
        all_alerts = []
        for p in patients:
             # Assuming db method supports patient_id filter or we get all
             p_alerts = db.get_unacknowledged_alerts(p['id']) 
             all_alerts.extend(p_alerts)

        return render_template('dashboard_caregiver.html', caregiver=caregiver, patients=patients, alerts=all_alerts)
        
    elif role == 'patient':
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM patients WHERE user_id = ?", (user_id,))
        patient = cursor.fetchone()
        
        if not patient:
            return "Patient profile not found", 404
            
        patient_id = patient['id']
        reminders = db.get_due_reminders(patient_id)
        active_zones = db.get_active_safety_zones(patient_id)
        
        return render_template('dashboard_patient.html', patient=patient, reminders=reminders, zones=active_zones)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- API Endpoints ---

@app.route('/api/dashboard/summary')
@login_required('caregiver')
def get_dashboard_summary():
    """Get dashboard summary data for polling"""
    try:
        user_id = session['user_id']
        cursor = db.connection.cursor()
        cursor.execute("SELECT id FROM caregivers WHERE user_id = ?", (user_id,))
        caregiver = cursor.fetchone()
        if not caregiver:
            return jsonify({'error': 'Caregiver not found'}), 404
            
        caregiver_id = caregiver['id']
        patients = db.get_caregiver_patients(caregiver_id)
        
        summary = {
            "patients": len(patients),
            "alerts": 0,
            "recent_logs": 0
        }
        
        all_alerts = []
        for p in patients:
             p_alerts = db.get_unacknowledged_alerts(p['id'])
             all_alerts.extend(p_alerts)
        
        summary['alerts'] = len(all_alerts)
        summary['timestamp'] = datetime.now().isoformat()
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
@login_required('caregiver')
def get_alerts():
    """Get all unacknowledged alerts for caregiver's patients"""
    user_id = session['user_id']
    cursor = db.connection.cursor()
    cursor.execute("SELECT id FROM caregivers WHERE user_id = ?", (user_id,))
    caregiver = cursor.fetchone()
    if not caregiver: 
        return jsonify([])
        
    caregiver_id = caregiver['id']
    patients = db.get_caregiver_patients(caregiver_id)
    all_alerts = []
    for p in patients:
            p_alerts = db.get_unacknowledged_alerts(p['id'])
            # Add patient name to alert for context
            for a in p_alerts:
                a['patient_name'] = p['name']
            all_alerts.extend(p_alerts)
            
    return jsonify(all_alerts)

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@login_required('caregiver')
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    # Verify caregiver has access to this alert's patient? 
    # For now, just acknowledge.
    success = db.acknowledge_alert(alert_id)
    if success:
        return jsonify({"status": "acknowledged"})
    else:
        return jsonify({"error": "Alert not found"}), 404

@app.route('/api/patient/<int:patient_id>/logs')
@login_required('caregiver')
def get_patient_logs(patient_id):
    logs = db.get_patient_logs(patient_id)
    return jsonify(logs)

@app.route('/api/patient/<int:patient_id>/reminders', methods=['GET', 'POST'])
@login_required()
def manage_reminders(patient_id):
    if request.method == 'POST':
        data = request.json
        title = data.get('title')
        r_type = data.get('type', 'medication')
        desc = data.get('description', '')
        time_str = data.get('scheduled_time') # ISO format
        repeat = data.get('repeat_pattern') # dict
        
        try:
            scheduled_time = datetime.fromisoformat(time_str)
        except ValueError:
            return jsonify({'error': 'Invalid time format'}), 400

        # Determine if created by caregiver or patient
        caregiver_id = None
        if session['role'] == 'caregiver':
            cursor = db.connection.cursor()
            cursor.execute("SELECT id FROM caregivers WHERE user_id = ?", (session['user_id'],))
            row = cursor.fetchone()
            if row:
                caregiver_id = row['id']
        
        rid = db.add_reminder(patient_id, r_type, title, desc, scheduled_time, repeat, caregiver_id)
        return jsonify({'id': rid, 'status': 'success'})
        
    else:
        # get all reminders including future ones
        # Use a method that gets all active reminders
        cursor = db.connection.cursor()
        cursor.execute("SELECT * FROM reminders WHERE patient_id = ? AND is_active = 1", (patient_id,))
        reminders = [dict(row) for row in cursor.fetchall()]
        return jsonify(reminders)

@app.route('/api/patient/<int:patient_id>/location')
@login_required('caregiver')
def get_patient_location(patient_id):
    # Retrieve latest location from logs
    logs = db.get_patient_logs(patient_id, hours_back=1)
    for log in logs:
        if log.get('location_lat') and log.get('location_lon'):
            return jsonify({
                'lat': log['location_lat'],
                'lon': log['location_lon'],
                'timestamp': log['timestamp']
            })
    return jsonify({'lat': None, 'lon': None})

@app.route('/api/patient/<int:patient_id>/known_persons', methods=['GET', 'POST'])
@login_required()
def manage_known_persons(patient_id):
    if request.method == 'POST':
        name = request.form['name']
        relationship = request.form['relationship']
        file = request.files.get('image')
        
        photo_path = None
        if file and file.filename:
            filename = secure_filename(file.filename)
            # Create patient specific folder
            p_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(patient_id))
            os.makedirs(p_folder, exist_ok=True)
            photo_path = os.path.join(p_folder, filename)
            file.save(photo_path)
        
        pid = db.add_known_person(patient_id, name, relationship, photo_path)
        return jsonify({'id': pid, 'status': 'success'})
        
    else:
        persons = db.get_all_known_persons(patient_id)
        return jsonify(persons)

@app.route('/api/caregiver/link_patient', methods=['POST'])
@login_required('caregiver')
def link_patient():
    """Link an existing patient to the current caregiver"""
    try:
        user_id = session['user_id']
        cursor = db.connection.cursor()
        cursor.execute("SELECT id FROM caregivers WHERE user_id = ?", (user_id,))
        caregiver = cursor.fetchone()
        
        if not caregiver:
            return jsonify({'error': 'Caregiver not found'}), 404
        
        caregiver_id = caregiver['id']
        data = request.json
        patient_id = data.get('patient_id')
        relationship_type = data.get('relationship_type', 'Primary')
        
        if not patient_id:
            return jsonify({'error': 'Patient ID required'}), 400
        
        # Check if patient exists
        cursor.execute("SELECT id, name FROM patients WHERE id = ?", (patient_id,))
        patient = cursor.fetchone()
        
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Link patient to caregiver
        try:
            cursor.execute("""
                INSERT INTO patient_caregivers (patient_id, caregiver_id, relationship_type, permissions)
                VALUES (?, ?, ?, ?)
            """, (patient_id, caregiver_id, relationship_type, json.dumps({'view': True, 'edit': True})))
            db.connection.commit()
            
            return jsonify({
                'status': 'success',
                'patient_id': patient_id,
                'patient_name': patient['name']
            })
        except Exception as e:
            db.connection.rollback()
            if 'UNIQUE constraint failed' in str(e):
                return jsonify({'error': 'Patient already linked to this caregiver'}), 400
            raise
            
    except Exception as e:
        logging.error(f"Error linking patient: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients/available', methods=['GET'])
@login_required('caregiver')
def get_available_patients():
    """Get list of all patients for linking"""
    try:
        cursor = db.connection.cursor()
        cursor.execute("SELECT id, name, details FROM patients ORDER BY name")
        patients = [dict(row) for row in cursor.fetchall()]
        return jsonify(patients)
    except Exception as e:
        logging.error(f"Error fetching patients: {e}")
        return jsonify({'error': str(e)}), 500

# ===== Patient Profile Management =====

@app.route('/api/patient/<int:patient_id>/profile', methods=['GET'])
@login_required()
def get_patient_profile(patient_id):
    """Get patient profile"""
    try:
        profile = db.get_patient_profile(patient_id)
        if profile:
            return jsonify(profile)
        return jsonify({'error': 'Patient not found'}), 404
    except Exception as e:
        logging.error(f"Error getting patient profile: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/profile', methods=['PUT'])
@login_required()
def update_patient_profile(patient_id):
    """Update patient profile"""
    try:
        data = request.json
        success = db.update_patient_profile(patient_id, data)
        if success:
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Failed to update profile'}), 400
    except Exception as e:
        logging.error(f"Error updating patient profile: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/profile/photo', methods=['POST'])
@login_required()
def upload_patient_photo(patient_id):
    """Upload patient profile photo"""
    try:
        file = request.files.get('photo')
        if not file or not file.filename:
            return jsonify({'error': 'No file provided'}), 400
        
        filename = secure_filename(file.filename)
        # Create patient-specific folder
        patient_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'profiles', str(patient_id))
        os.makedirs(patient_folder, exist_ok=True)
        
        # Save with timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        photo_filename = f"profile_{timestamp}_{filename}"
        photo_path = os.path.join(patient_folder, photo_filename)
        file.save(photo_path)
        
        # Update database
        success = db.update_patient_photo(patient_id, photo_path)
        if success:
            return jsonify({'status': 'success', 'photo_path': photo_path})
        return jsonify({'error': 'Failed to update photo'}), 400
    except Exception as e:
        logging.error(f"Error uploading patient photo: {e}")
        return jsonify({'error': str(e)}), 500

# ===== Safety Zone Management =====

@app.route('/api/patient/<int:patient_id>/safety_zones', methods=['GET'])
@login_required()
def get_safety_zones(patient_id):
    """Get all safety zones for a patient"""
    try:
        zones = db.get_safety_zones(patient_id)
        return jsonify(zones)
    except Exception as e:
        logging.error(f"Error getting safety zones: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/safety_zones', methods=['POST'])
@login_required('caregiver')
def create_safety_zone(patient_id):
    """Create a new safety zone"""
    try:
        data = request.json
        zone_id = db.add_safety_zone(
            patient_id,
            data.get('name'),
            data.get('center_lat'),
            data.get('center_lon'),
            data.get('radius_meters'),
            data.get('zone_type', 'safe')
        )
        if zone_id > 0:
            return jsonify({'id': zone_id, 'status': 'success'})
        return jsonify({'error': 'Failed to create zone'}), 400
    except Exception as e:
        logging.error(f"Error creating safety zone: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/known_persons', methods=['GET', 'POST'])
@login_required
def manage_known_persons(patient_id):
    if request.method == 'GET':
        persons = db.get_known_persons(patient_id)
        return jsonify(persons)
        
    elif request.method == 'POST':
        # Handle form data (multipart/form-data)
        name = request.form.get('name')
        relationship = request.form.get('relationship')
        phone = request.form.get('phone')
        email = request.form.get('email')
        notes = request.form.get('notes')
        is_emergency = request.form.get('is_emergency_contact') == 'true'
        
        file = request.files.get('image')
        
        if not file:
            return jsonify({'error': 'No image file provided'}), 400
            
        filename = secure_filename(file.filename)
        # Save to backend/static/uploads/known_persons/<patient_id>/
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'known_persons', str(patient_id))
        os.makedirs(upload_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        photo_filename = f"{name}_{timestamp}_{filename}"
        photo_path = os.path.join(upload_folder, photo_filename)
        file.save(photo_path)
        
        # Save to DB (Assuming db methods exist)
        try:
            # Need to implement add_known_person in db_manager if not exists
            # For now assume it exists or needs to be added
            success = db.add_known_person(patient_id, name, relationship, photo_path, phone, email, notes, is_emergency)
            if success:
                return jsonify({'message': 'Known person added successfully', 'photo_path': photo_path})
            else:
                return jsonify({'error': 'Database error'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
@app.route('/api/patient/<int:patient_id>/safety_zones/<int:zone_id>', methods=['PUT'])
@login_required('caregiver')
def update_safety_zone(patient_id, zone_id):
    """Update a safety zone"""
    try:
        data = request.json
        success = db.update_safety_zone(zone_id, data)
        if success:
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Failed to update zone'}), 400
    except Exception as e:
        logging.error(f"Error updating safety zone: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/safety_zones/<int:zone_id>', methods=['DELETE'])
@login_required('caregiver')
def delete_safety_zone(patient_id, zone_id):
    """Delete a safety zone"""
    try:
        success = db.delete_safety_zone(zone_id)
        if success:
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Failed to delete zone'}), 400
    except Exception as e:
        logging.error(f"Error deleting safety zone: {e}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
