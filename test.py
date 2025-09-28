"""
Memory System Debug Script
Checks the agentic therapy system memory without running the full Flask app
"""

import os
import sys
import json
import sqlite3
from datetime import datetime

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agentic_therapy_system import AgenticTherapySystem, UserMemoryManager, EncryptionManager
    print("✓ Successfully imported agentic modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def check_database_exists():
    """Check if the database file exists and what tables it has"""
    db_path = "user_memory.db"
    
    print("\n" + "="*60)
    print("DATABASE ANALYSIS")
    print("="*60)
    
    if not os.path.exists(db_path):
        print("❌ Database file 'user_memory.db' does not exist")
        return False
    
    print(f"✓ Database file exists: {db_path}")
    print(f"  Size: {os.path.getsize(db_path)} bytes")
    print(f"  Modified: {datetime.fromtimestamp(os.path.getmtime(db_path))}")
    
    # Check tables
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\n📊 Tables in database:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} records")
            
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                sample = cursor.fetchone()
                print(f"    Sample data length: {len(str(sample)) if sample else 0} chars")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False

def test_encryption():
    """Test encryption/decryption functionality"""
    print("\n" + "="*60)
    print("ENCRYPTION TEST")
    print("="*60)
    
    try:
        # Test encryption manager
        encryption_manager = EncryptionManager()
        test_password = "test_password_123"
        encryption_manager.set_password(test_password)
        
        test_data = "This is test conversation data"
        encrypted = encryption_manager.encrypt_data(test_data)
        decrypted = encryption_manager.decrypt_data(encrypted)
        
        print(f"✓ Encryption test passed")
        print(f"  Original: {test_data}")
        print(f"  Encrypted: {encrypted[:50]}...")
        print(f"  Decrypted: {decrypted}")
        print(f"  Match: {test_data == decrypted}")
        
        return True
        
    except Exception as e:
        print(f"❌ Encryption test failed: {e}")
        return False

def test_user_memory_manager():
    """Test the user memory manager functionality"""
    print("\n" + "="*60)
    print("USER MEMORY MANAGER TEST")
    print("="*60)
    
    try:
        memory_manager = UserMemoryManager()
        
        # Test user creation
        test_user_id = "debug_user_123"
        test_password = "debug_password_456"
        test_name = "Debug User"
        
        print(f"Creating test user: {test_user_id}")
        success = memory_manager.create_user(test_user_id, test_password, test_name)
        print(f"User creation: {'✓ Success' if success else '❌ Failed'}")
        
        if success:
            # Test user authentication
            print(f"Testing authentication...")
            auth_success = memory_manager.authenticate_user(test_user_id, test_password)
            print(f"Authentication: {'✓ Success' if auth_success else '❌ Failed'}")
            
            if auth_success:
                # Test profile retrieval
                profile = memory_manager.get_user_profile(test_user_id)
                print(f"Profile retrieval: {'✓ Success' if profile else '❌ Failed'}")
                
                if profile:
                    print(f"  User ID: {profile.user_id}")
                    print(f"  Name: {profile.preferred_name}")
                    print(f"  Last activity: {profile.last_activity}")
                
                # Test session history
                sessions = memory_manager.get_recent_sessions(test_user_id, limit=5)
                print(f"Recent sessions: {len(sessions)} found")
                
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ User memory manager test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_agentic_system():
    """Test the full agentic system"""
    print("\n" + "="*60)
    print("AGENTIC SYSTEM TEST")
    print("="*60)
    
    try:
        agentic_system = AgenticTherapySystem()
        
        # Test system initialization
        print(f"✓ Agentic system initialized")
        print(f"  Privacy mode: {agentic_system.privacy_mode}")
        print(f"  Current user: {agentic_system.current_user_id}")
        print(f"  Groq API key: {'✓ Configured' if agentic_system.groq_api_key else '❌ Missing'}")
        
        # Test privacy consent simulation
        print(f"\nTesting privacy consent flow...")
        consent_request = agentic_system.request_privacy_consent()
        print(f"Consent request generated: {'✓ Success' if consent_request else '❌ Failed'}")
        
        # Simulate user choosing agentic mode
        user_choice = {
            "choice": "remember",
            "retention_days": 30,
            "preferred_name": "Test User",
            "password": "test_password_789"
        }
        
        consent_result = agentic_system.handle_privacy_consent(user_choice)
        print(f"Consent handling: {'✓ Success' if consent_result.get('status') == 'success' else '❌ Failed'}")
        
        if consent_result.get('status') == 'success':
            print(f"  User ID: {consent_result.get('user_id')}")
            print(f"  Agentic mode: {not agentic_system.privacy_mode}")
            print(f"  Current user set: {agentic_system.current_user_id}")
            
            # Test memory context building
            test_message = "Hello, I've been feeling anxious lately"
            test_analysis = {
                'dominant_emotion': 'anxiety',
                'sentiment': 'negative',
                'mental_health_topics': [('anxiety', 0.8)],
                'suggested_techniques': ['breathing', 'mindfulness']
            }
            
            print(f"\nTesting context building...")
            context = agentic_system.build_personalized_context(test_message, test_analysis)
            print(f"Context built: {'✓ Success' if context else '❌ Failed'}")
            print(f"  Personalization available: {context.get('personalization_available', False)}")
            print(f"  Recent sessions: {len(context.get('recent_sessions', []))}")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Agentic system test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def check_existing_users():
    """Check what users exist in the database"""
    print("\n" + "="*60)
    print("EXISTING USERS ANALYSIS")
    print("="*60)
    
    try:
        conn = sqlite3.connect("user_memory.db")
        cursor = conn.cursor()
        
        # Check user profiles
        cursor.execute("SELECT user_id, created_date, last_access FROM user_profiles")
        users = cursor.fetchall()
        
        print(f"Found {len(users)} users in database:")
        for user in users:
            user_id, created, last_access = user
            print(f"  - {user_id}")
            print(f"    Created: {created}")
            print(f"    Last access: {last_access}")
            
            # Check sessions for this user
            cursor.execute("SELECT COUNT(*) FROM session_summaries WHERE user_id = ?", (user_id,))
            session_count = cursor.fetchone()[0]
            print(f"    Sessions: {session_count}")
            
            # Check goals for this user
            cursor.execute("SELECT COUNT(*) FROM user_goals WHERE user_id = ?", (user_id,))
            goal_count = cursor.fetchone()[0]
            print(f"    Goals: {goal_count}")
            print()
        
        conn.close()
        return len(users) > 0
        
    except Exception as e:
        print(f"❌ Failed to check existing users: {e}")
        return False

def test_memory_integration():
    """Test if memory integration is working for existing users"""
    print("\n" + "="*60)
    print("MEMORY INTEGRATION TEST")
    print("="*60)
    
    try:
        # Try to load existing users and test their memory
        memory_manager = UserMemoryManager()
        
        conn = sqlite3.connect("user_memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM user_profiles LIMIT 1")
        result = cursor.fetchone()
        
        if not result:
            print("❌ No existing users found to test")
            return False
        
        user_id = result[0]
        print(f"Testing memory for user: {user_id}")
        
        # Try to get recent sessions (this should work without authentication for testing)
        try:
            # We need to simulate authentication to test memory
            # This is tricky without the actual password
            print("❌ Cannot test memory without user password")
            print("   Memory testing requires authentication")
            return False
            
        except Exception as e:
            print(f"❌ Memory integration test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Memory integration test failed: {e}")
        return False

def main():
    """Run all debug tests"""
    print("AGENTIC THERAPY SYSTEM - MEMORY DEBUG")
    print("="*60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print()
    
    results = {}
    
    # Run tests
    results['database'] = check_database_exists()
    results['encryption'] = test_encryption()
    results['user_manager'] = test_user_memory_manager()
    results['existing_users'] = check_existing_users()
    results['agentic_system'] = test_agentic_system()
    
    # Summary
    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Memory system should be working.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed. Memory system has issues.")
        
        # Provide specific guidance
        if not results['database']:
            print("\n💡 No database found. The system hasn't stored any user data yet.")
        elif not results['encryption']:
            print("\n💡 Encryption failed. Check if cryptography library is installed.")
        elif not results['user_manager']:
            print("\n💡 User management failed. Database schema or encryption issues.")
        elif not results['agentic_system']:
            print("\n💡 Agentic system failed. Check config.json and dependencies.")

if __name__ == "__main__":
    main()