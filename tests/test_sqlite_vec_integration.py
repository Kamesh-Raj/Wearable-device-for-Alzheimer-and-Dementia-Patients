"""
Test script for sqlite-vec integration
Demonstrates vector similarity search for face recognition
"""

import numpy as np
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

try:
    from database.db_manager import DatabaseManager, SQLITE_VEC_AVAILABLE
    from dashboard.backend.database import DatabaseManager as DashboardDB
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_database_manager():
    """Test the main database manager with sqlite-vec"""
    print("\n=== Testing Database Manager (database/db_manager.py) ===")
    
    if not SQLITE_VEC_AVAILABLE:
        print("❌ sqlite-vec is not installed!")
        print("Install it with: pip install sqlite-vec")
        return False
    
    print("✓ sqlite-vec is available")
    
    # Initialize database
    db = DatabaseManager("data/test_neuro_assistive.db")
    print("✓ Database initialized")
    
    # Add known persons
    # Need to register a patient first
    user_id = db.create_user("testuser_vec", "testpass_vec", "patient")
    patient_id = db.register_patient("Test Patient Vec", user_id=user_id)
    print(f"✓ Registered patient: {patient_id}")

    person1_id = db.add_known_person(
        patient_id=patient_id,
        name="John Doe",
        relationship="Family",
        notes="Patient's son"
    )
    print(f"✓ Added person 1: ID={person1_id}")
    
    person2_id = db.add_known_person(
        patient_id=patient_id,
        name="Jane Smith",
        relationship="Caregiver",
        notes="Primary caregiver"
    )
    print(f"✓ Added person 2: ID={person2_id}")
    
    # Generate sample face embeddings (512-dimensional)
    embedding1 = np.random.rand(512).astype(np.float32)
    embedding2 = np.random.rand(512).astype(np.float32)
    
    # Add embeddings
    success1 = db.add_face_embedding(person1_id, patient_id, embedding1, "images/john_1.jpg")
    success2 = db.add_face_embedding(person2_id, patient_id, embedding2, "images/jane_1.jpg")
    
    if success1 and success2:
        print("✓ Added face embeddings")
    else:
        print("❌ Failed to add embeddings")
        return False
    
    # Search for similar faces
    query_embedding = embedding1 + np.random.rand(512).astype(np.float32) * 0.1
    results = db.search_face_embedding(patient_id, query_embedding, n_results=5)
    
    print(f"\n✓ Search results: {len(results)} matches found")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['person_name']} ({result['relationship']})")
        print(f"     Similarity: {result['similarity']:.4f}, Distance: {result['distance']:.4f}")
    
    # Get all known persons
    persons = db.get_all_known_persons(patient_id)
    print(f"\n✓ Total known persons: {len(persons)}")
    
    db.close()
    return True


def test_dashboard_database():
    """Test the dashboard database manager with sqlite-vec"""
    print("\n=== Testing Dashboard Database (dashboard/backend/database.py) ===")
    
    if not SQLITE_VEC_AVAILABLE:
        print("❌ sqlite-vec is not installed!")
        return False
    
    print("✓ sqlite-vec is available")
    
    # Initialize database
    db = DashboardDB("dashboard/data/test_sentient_edge.db")
    print("✓ Database initialized")
    
    # Add a patient
    patient_id = db.add_patient(
        name="Alice Johnson",
        age=78,
        gender="Female",
        medical_conditions="Alzheimer's Disease",
        emergency_contact="Bob Johnson",
        emergency_phone="+1-555-0123"
    )
    print(f"✓ Added patient: {patient_id}")
    
    # Add a known face
    face_id = db.add_known_face(
        patient_id=patient_id,
        name="Bob Johnson",
        relationship="Husband"
    )
    print(f"✓ Added known face: {face_id}")
    
    # Generate sample face embedding
    embedding = np.random.rand(512).astype(np.float32).tolist()
    
    # Add face embedding
    db.add_face_embedding(face_id, embedding, "images/bob_1.jpg")
    print("✓ Added face embedding")
    
    # Search for similar faces
    query_embedding = np.array(embedding) + np.random.rand(512).astype(np.float32) * 0.1
    results = db.search_face(query_embedding.tolist(), n_results=5)
    
    print(f"\n✓ Search results: {len(results)} matches found")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Face ID: {result['face_id']}")
        print(f"     Similarity: {result['similarity']:.4f}, Distance: {result['distance']:.4f}")
    
    # Get embedding count
    count = db.get_face_embeddings_count(face_id)
    print(f"\n✓ Total embeddings for face {face_id}: {count}")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("SQLite-Vec Integration Test")
    print("=" * 60)
    
    if not SQLITE_VEC_AVAILABLE:
        print("\n❌ CRITICAL: sqlite-vec is not installed!")
        print("\nTo install, run:")
        print("  pip install sqlite-vec")
        print("\nOr install from requirements:")
        print("  pip install -r requirements_sqlite_vec.txt")
        return
    
    # Test both database managers
    test1_passed = test_database_manager()
    test2_passed = test_dashboard_database()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Database Manager: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Dashboard Database: {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n❌ Some tests failed")


if __name__ == "__main__":
    main()
