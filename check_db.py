import sqlite3
from datetime import datetime

# Connect to database
conn = sqlite3.connect('user_memory.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("=" * 60)
print("DATABASE STRUCTURE")
print("=" * 60)
print(f"Tables: {[t[0] for t in tables]}\n")

# Check user_profiles
cursor.execute('SELECT user_id, created_date, last_access, retention_days FROM user_profiles')
users = cursor.fetchall()
print(f"USER PROFILES ({len(users)} users):")
print("-" * 60)
for user in users:
    print(f"  User ID: {user[0]}")
    print(f"  Created: {user[1]}")
    print(f"  Last Access: {user[2]}")
    print(f"  Retention Days: {user[3]}")
    print()

# Check session_summaries
cursor.execute('SELECT session_id, user_id, session_date FROM session_summaries')
sessions = cursor.fetchall()
print(f"\nSESSION SUMMARIES ({len(sessions)} sessions):")
print("-" * 60)
for session in sessions:
    print(f"  Session: {session[0]}")
    print(f"  User: {session[1]}")
    print(f"  Date: {session[2]}")
    print()

# Check user_goals
cursor.execute('SELECT goal_id, user_id, status, created_date FROM user_goals')
goals = cursor.fetchall()
print(f"\nUSER GOALS ({len(goals)} goals):")
print("-" * 60)
for goal in goals:
    print(f"  Goal ID: {goal[0]}")
    print(f"  User: {goal[1]}")
    print(f"  Status: {goal[2]}")
    print(f"  Created: {goal[3]}")
    print()

# Check privacy_consents
cursor.execute('SELECT user_id, memory_consent, retention_days, consent_date FROM privacy_consents')
consents = cursor.fetchall()
print(f"\nPRIVACY CONSENTS ({len(consents)} records):")
print("-" * 60)
for consent in consents:
    print(f"  User: {consent[0]}")
    print(f"  Memory Consent: {bool(consent[1])}")
    print(f"  Retention Days: {consent[2]}")
    print(f"  Consent Date: {consent[3]}")
    print()

conn.close()

print("=" * 60)
print("DATABASE CHECK COMPLETE")
print("=" * 60)
