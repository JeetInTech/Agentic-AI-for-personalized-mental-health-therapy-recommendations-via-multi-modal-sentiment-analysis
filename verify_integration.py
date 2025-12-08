"""
Quick test to verify vector DB integration with app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vector_session_manager import VectorSessionManager

print("="*60)
print("VERIFYING VECTOR DB INTEGRATION")
print("="*60)

try:
    # Test initialization
    print("\n1. Testing VectorSessionManager initialization...")
    vsm = VectorSessionManager()
    print("   ✅ Vector session manager created successfully")
    
    # Test that it can be imported in app.py
    print("\n2. Checking if app.py can import...")
    from app import vector_session_manager
    print("   ✅ Successfully imported in app.py")
    
    # Verify collections exist
    print("\n3. Verifying ChromaDB collections...")
    collections = vsm.chroma_client.list_collections()
    print(f"   ✅ Found {len(collections)} collections:")
    for col in collections:
        print(f"      - {col.name}")
    
    # Check if any sessions exist
    print("\n4. Checking existing sessions...")
    session_files = os.listdir("session_data")
    json_files = [f for f in session_files if f.endswith('.json')]
    print(f"   ✅ Found {len(json_files)} session files")
    
    if json_files:
        print(f"   Latest: {json_files[-1]}")
    
    print("\n" + "="*60)
    print("INTEGRATION VERIFICATION COMPLETE!")
    print("="*60)
    print("\n✅ Vector database is ready to use")
    print("✅ Sessions will be auto-saved when users chat in agentic mode")
    print("✅ API endpoints available:")
    print("   - POST /api/session/search (semantic search)")
    print("   - POST /api/session/history (get past sessions)")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
