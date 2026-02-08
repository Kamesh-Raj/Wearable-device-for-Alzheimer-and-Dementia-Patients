"""
Safety & Management Module
Handles GPS tracking, geofencing, and caregiver alerts
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

class SafetyMonitor:
    """Monitors patient safety through GPS tracking and geofencing"""
    
    def __init__(self, db_manager, patient_id: int):
        self.logger = logging.getLogger(__name__)
        self.db = db_manager
        self.patient_id = patient_id
        
        # Safety configuration
        self.location_history = []
        self.max_history_size = 100
        self.alert_cooldown_minutes = 15  # Prevent spam alerts
        self.last_alert_times = {}
        
        # Movement patterns
        self.movement_threshold_meters = 50  # Minimum movement to consider significant
        self.stationary_threshold_minutes = 60  # Alert if stationary too long
        
    async def check_safety(self, location: Dict[str, float]) -> Dict[str, Any]:
        """Check current location against safety zones and patterns"""
        try:
            if not location:
                return {'is_safe': False, 'reason': 'no_gps_signal'}
            
            lat = location['latitude']
            lon = location['longitude']
            timestamp = location.get('timestamp', datetime.now().timestamp())
            
            # Add to location history
            self._add_location_to_history(lat, lon, timestamp)
            
            # Check geofencing
            zone_check = self.db.check_location_safety(self.patient_id, lat, lon)
            
            # Check movement patterns
            movement_check = self._check_movement_patterns()
            
            # Check for wandering behavior
            wandering_check = self._check_wandering_patterns()
            
            # Combine all safety checks
            safety_status = self._evaluate_overall_safety(
                zone_check, movement_check, wandering_check, location
            )
            
            # Create alerts if necessary
            if not safety_status['is_safe']:
                await self._handle_safety_alert(safety_status)
            
            return safety_status
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return {'is_safe': False, 'reason': 'system_error', 'error': str(e)}
    
    def _add_location_to_history(self, lat: float, lon: float, timestamp: float):
        """Add location point to history buffer"""
        try:
            location_point = {
                'latitude': lat,
                'longitude': lon,
                'timestamp': timestamp
            }
            
            self.location_history.append(location_point)
            
            # Maintain history size limit
            if len(self.location_history) > self.max_history_size:
                self.location_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Failed to add location to history: {e}")
    
    def _check_movement_patterns(self) -> Dict[str, Any]:
        """Analyze recent movement patterns for anomalies"""
        try:
            if len(self.location_history) < 2:
                return {'status': 'insufficient_data'}
            
            recent_locations = self.location_history[-10:]  # Last 10 locations
            current_time = datetime.now().timestamp()
            
            # Check if patient has been stationary too long
            if len(recent_locations) >= 5:
                distances = []
                for i in range(1, len(recent_locations)):
                    dist = self._calculate_distance(
                        recent_locations[i-1]['latitude'], recent_locations[i-1]['longitude'],
                        recent_locations[i]['latitude'], recent_locations[i]['longitude']
                    )
                    distances.append(dist)
                
                avg_movement = sum(distances) / len(distances)
                time_span = current_time - recent_locations[0]['timestamp']
                
                # Check for concerning patterns
                if avg_movement < 10 and time_span > self.stationary_threshold_minutes * 60:
                    return {
                        'status': 'stationary_too_long',
                        'duration_minutes': time_span / 60,
                        'average_movement_meters': avg_movement
                    }
                
                # Check for rapid movement (possible distress)
                if avg_movement > 200:  # Moving very fast
                    return {
                        'status': 'rapid_movement',
                        'average_speed_mps': avg_movement / (time_span / len(distances))
                    }
            
            return {'status': 'normal'}
            
        except Exception as e:
            self.logger.error(f"Movement pattern check failed: {e}")
            return {'status': 'error'}
    
    def _check_wandering_patterns(self) -> Dict[str, Any]:
        """Check for wandering behavior patterns"""
        try:
            if len(self.location_history) < 10:
                return {'status': 'insufficient_data'}
            
            recent_locations = self.location_history[-20:]  # Last 20 locations
            
            # Calculate path characteristics
            total_distance = 0
            displacement = 0
            
            for i in range(1, len(recent_locations)):
                dist = self._calculate_distance(
                    recent_locations[i-1]['latitude'], recent_locations[i-1]['longitude'],
                    recent_locations[i]['latitude'], recent_locations[i]['longitude']
                )
                total_distance += dist
            
            # Calculate displacement (straight-line distance from start to end)
            if len(recent_locations) >= 2:
                displacement = self._calculate_distance(
                    recent_locations[0]['latitude'], recent_locations[0]['longitude'],
                    recent_locations[-1]['latitude'], recent_locations[-1]['longitude']
                )
            
            # Calculate tortuosity (path efficiency)
            tortuosity = total_distance / max(displacement, 1)  # Avoid division by zero
            
            # High tortuosity indicates wandering/circling behavior
            if tortuosity > 3.0 and total_distance > 100:
                return {
                    'status': 'possible_wandering',
                    'tortuosity': tortuosity,
                    'total_distance': total_distance,
                    'displacement': displacement
                }
            
            return {'status': 'normal', 'tortuosity': tortuosity}
            
        except Exception as e:
            self.logger.error(f"Wandering pattern check failed: {e}")
            return {'status': 'error'}
    
    def _evaluate_overall_safety(self, zone_check: Dict[str, Any], 
                               movement_check: Dict[str, Any],
                               wandering_check: Dict[str, Any],
                               location: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate overall safety status from all checks"""
        try:
            safety_issues = []
            risk_level = 0
            
            # Check zone safety
            if not zone_check.get('is_safe', False):
                safety_issues.append('outside_safe_zone')
                risk_level += 3
                
                zone = zone_check.get('zone')
                if zone and zone.get('zone_type') == 'restricted':
                    risk_level += 2
            
            # Check movement patterns
            movement_status = movement_check.get('status', 'normal')
            if movement_status == 'stationary_too_long':
                safety_issues.append('stationary_too_long')
                risk_level += 2
            elif movement_status == 'rapid_movement':
                safety_issues.append('rapid_movement')
                risk_level += 2
            
            # Check wandering patterns
            wandering_status = wandering_check.get('status', 'normal')
            if wandering_status == 'possible_wandering':
                safety_issues.append('possible_wandering')
                risk_level += 3
            
            # Determine overall safety
            is_safe = risk_level == 0
            
            # Determine alert priority
            if risk_level >= 5:
                priority = 'critical'
            elif risk_level >= 3:
                priority = 'high'
            elif risk_level >= 1:
                priority = 'medium'
            else:
                priority = 'low'
            
            return {
                'is_safe': is_safe,
                'risk_level': risk_level,
                'priority': priority,
                'safety_issues': safety_issues,
                'zone_check': zone_check,
                'movement_check': movement_check,
                'wandering_check': wandering_check,
                'location': location,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Safety evaluation failed: {e}")
            return {
                'is_safe': False,
                'risk_level': 5,
                'priority': 'critical',
                'safety_issues': ['system_error'],
                'error': str(e)
            }
    
    async def _handle_safety_alert(self, safety_status: Dict[str, Any]):
        """Handle safety alerts with appropriate escalation"""
        try:
            priority = safety_status.get('priority', 'medium')
            safety_issues = safety_status.get('safety_issues', [])
            
            # Check alert cooldown to prevent spam
            alert_key = f"{priority}_{'-'.join(safety_issues)}"
            current_time = datetime.now()
            
            if alert_key in self.last_alert_times:
                time_since_last = current_time - self.last_alert_times[alert_key]
                if time_since_last.total_seconds() < self.alert_cooldown_minutes * 60:
                    return  # Skip alert due to cooldown
            
            # Create alert message
            alert_message = self._create_alert_message(safety_status)
            
            # Create database alert
            alert_id = self.db.create_alert(
                patient_id=self.patient_id,
                alert_type='safety',
                message=alert_message,
                priority=self._get_priority_level(priority),
                data=safety_status
            )
            
            # Log safety event
            self.db.log_patient_data(
                patient_id=self.patient_id,
                log_type='safety',
                data=safety_status,
                severity_level=safety_status.get('risk_level', 0),
                location=(
                    safety_status['location']['latitude'],
                    safety_status['location']['longitude']
                ) if safety_status.get('location') else None
            )
            
            # Update alert cooldown
            self.last_alert_times[alert_key] = current_time
            
            self.logger.warning(f"Safety alert created: {alert_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle safety alert: {e}")
    
    def _create_alert_message(self, safety_status: Dict[str, Any]) -> str:
        """Create human-readable alert message"""
        try:
            safety_issues = safety_status.get('safety_issues', [])
            location = safety_status.get('location', {})
            priority = safety_status.get('priority', 'medium')
            
            if 'outside_safe_zone' in safety_issues:
                zone_info = safety_status.get('zone_check', {}).get('zone', {})
                zone_name = zone_info.get('name', 'unknown area')
                return f"Patient has left the safe zone and is now in {zone_name}"
            
            elif 'possible_wandering' in safety_issues:
                wandering_info = safety_status.get('wandering_check', {})
                tortuosity = wandering_info.get('tortuosity', 0)
                return f"Patient showing wandering behavior (path efficiency: {tortuosity:.1f})"
            
            elif 'stationary_too_long' in safety_issues:
                movement_info = safety_status.get('movement_check', {})
                duration = movement_info.get('duration_minutes', 0)
                return f"Patient has been stationary for {duration:.0f} minutes"
            
            elif 'rapid_movement' in safety_issues:
                return "Patient is moving unusually fast - possible distress"
            
            else:
                return f"Safety concern detected (priority: {priority})"
                
        except Exception as e:
            self.logger.error(f"Failed to create alert message: {e}")
            return "Safety alert - please check patient status"
    
    def _get_priority_level(self, priority_str: str) -> int:
        """Convert priority string to numeric level"""
        priority_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        return priority_map.get(priority_str, 2)
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        from math import radians, sin, cos, sqrt, atan2
        
        # Earth's radius in meters
        R = 6371000
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = (sin(delta_lat / 2) ** 2 + 
             cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    
    def add_safe_zone(self, name: str, center_lat: float, center_lon: float,
                     radius_meters: int) -> int:
        """Add a new safe zone"""
        try:
            zone_id = self.db.add_safety_zone(
                patient_id=self.patient_id,
                name=name,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_meters=radius_meters,
                zone_type='safe'
            )
            
            if zone_id > 0:
                self.logger.info(f"Added safe zone: {name}")
            
            return zone_id
            
        except Exception as e:
            self.logger.error(f"Failed to add safe zone: {e}")
            return -1
    
    def get_location_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get recent location history"""
        try:
            cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
            
            recent_history = [
                loc for loc in self.location_history 
                if loc['timestamp'] >= cutoff_time
            ]
            
            return recent_history
            
        except Exception as e:
            self.logger.error(f"Failed to get location history: {e}")
            return []
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get summary of current safety status"""
        try:
            if not self.location_history:
                return {'status': 'no_location_data'}
            
            latest_location = self.location_history[-1]
            zone_check = self.db.check_location_safety(
                self.patient_id,
                latest_location['latitude'], 
                latest_location['longitude']
            )
            
            return {
                'status': 'active',
                'current_location': latest_location,
                'in_safe_zone': zone_check.get('is_safe', False),
                'zone_info': zone_check.get('zone'),
                'location_history_count': len(self.location_history),
                'last_update': datetime.fromtimestamp(latest_location['timestamp']).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get safety summary: {e}")
            return {'status': 'error', 'error': str(e)}