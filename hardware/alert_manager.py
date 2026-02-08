"""
Alert Manager - Hardware Bridge to Backend API
Handles sending alerts (gait, safe zone, distress) and receiving reminders.
"""

import requests
import json
import logging
import time
import geopy.distance # Optional for local geofencing check

class AlertManager:
    def __init__(self, api_base_url="http://127.0.0.1:5000/api", patient_id=1):
        self.logger = logging.getLogger(__name__)
        self.base_url = api_base_url
        self.patient_id = patient_id
        self.last_safey_check = 0
        self.safe_zones = [] # Cache safe zones
        self.is_connected = False
        
        # Test connection
        self.check_connection()

    def check_connection(self):
        try:
            requests.get(f"{self.base_url}/health", timeout=5)
            self.is_connected = True
            self.logger.info("Connected to Backend API.")
        except Exception as e:
            self.logger.error(f"Failed to connect to backend: {e}")
            self.is_connected = False
            
    def send_alert(self, alert_type, message, severity="high", data=None):
        """
        Send alert to caregiver via backend.
        Backend handles Gmail/SMS routing.
        """
        payload = {
            "patient_id": self.patient_id,
            "type": alert_type,
            "message": message,
            "severity": severity,
            "data": data or {},
            "timestamp": time.time()
        }
        
        try:
            resp = requests.post(f"{self.base_url}/alerts", json=payload, timeout=5)
            if resp.status_code == 200:
                self.logger.info(f"Alert sent: {alert_type}")
                return True
            else:
                self.logger.error(f"Failed to send alert: {resp.text}")
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            
        return False

    def check_safe_zone(self, lat, lon):
        """
        Periodically fetch safe zones and check current location.
        """
        # Update zones every 5 mins or if empty
        if not self.safe_zones or (time.time() - self.last_safey_check > 300):
            self.fetch_safe_zones()
            
        is_safe = False
        current_zone_name = "Unknown"
        
        if not self.safe_zones:
            return True, "No zones defined" # Regard as safe if no zones

        for zone in self.safe_zones:
            zone_lat = zone['center_lat']
            zone_lon = zone['center_lon']
            radius = zone['radius_meters']
            
            # Simple Haversine distance
            dist = geopy.distance.geodesic((lat, lon), (zone_lat, zone_lon)).meters
            
            if dist <= radius:
                is_safe = True
                current_zone_name = zone['name']
                if zone['type'] == 'restricted':
                    is_safe = False # Explicitly restricted area
                    return False, f"Entered Restricted Zone: {current_zone_name}"
                break
        
        if not is_safe:
            return False, f"Out of Safe Zone (Nearest: {current_zone_name})"
            
        return True, f"In Safe Zone: {current_zone_name}"

    def fetch_safe_zones(self):
        """Get zones from API"""
        try:
            resp = requests.get(f"{self.base_url}/patient/{self.patient_id}/safety_zones")
            if resp.status_code == 200:
                self.safe_zones = resp.json()
                self.last_safey_check = time.time()
        except:
            pass

    def get_reminders(self):
        """Poll for active reminders"""
        try:
            resp = requests.get(f"{self.base_url}/patient/{self.patient_id}/reminders/active")
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return []
