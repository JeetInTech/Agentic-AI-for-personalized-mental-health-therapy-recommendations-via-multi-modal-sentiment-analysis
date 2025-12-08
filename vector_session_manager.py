"""
Vector Database Session Manager
Stores sessions in JSON + ChromaDB for semantic search and retrieval
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid

logger = logging.getLogger(__name__)

class VectorSessionManager:
    """
    Manages therapy sessions with:
    1. JSON file storage for easy access
    2. ChromaDB for semantic search of past conversations
    """
    
    def __init__(self, 
                 sessions_dir: str = "session_data",
                 vector_db_dir: str = "vector_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.sessions_dir = sessions_dir
        self.vector_db_dir = vector_db_dir
        
        # Create directories
        os.makedirs(sessions_dir, exist_ok=True)
        os.makedirs(vector_db_dir, exist_ok=True)
        
        # Initialize ChromaDB with default embedding function
        self.chroma_client = chromadb.PersistentClient(
            path=vector_db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use ChromaDB's built-in default embedding function (doesn't require sentence-transformers)
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collections
        self.conversations_collection = self.chroma_client.get_or_create_collection(
            name="therapy_conversations",
            embedding_function=default_ef,
            metadata={"description": "User therapy conversation embeddings"}
        )
        
        self.insights_collection = self.chroma_client.get_or_create_collection(
            name="therapy_insights",
            embedding_function=default_ef,
            metadata={"description": "Key insights and patterns from sessions"}
        )
        
        logger.info("Vector session manager initialized with ChromaDB default embeddings")
    
    def save_session(self, 
                     user_id: str,
                     session_id: str,
                     session_data: Dict[str, Any]) -> bool:
        """
        Save session to both JSON file and vector database
        """
        try:
            # 1. Save to JSON file
            session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
            
            # Add metadata
            session_data['user_id'] = user_id
            session_data['session_id'] = session_id
            session_data['saved_at'] = datetime.now().isoformat()
            
            # Convert datetime objects to ISO format strings
            session_data_copy = self._serialize_datetime(session_data)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data_copy, f, indent=2, ensure_ascii=False, default=str)
            
            # 2. Index conversations in vector database
            self._index_session_conversations(user_id, session_id, session_data)
            
            logger.info(f"✅ Session {session_id} saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    def _index_session_conversations(self, 
                                     user_id: str,
                                     session_id: str,
                                     session_data: Dict[str, Any]):
        """
        Index individual messages and insights for semantic search
        """
        try:
            chat_history = session_data.get('chat_history', [])
            
            # Index each user message with assistant response
            for i, message in enumerate(chat_history):
                if message.get('role') == 'user':
                    user_text = message.get('content', '')
                    
                    # Get assistant response if available
                    assistant_response = ''
                    if i + 1 < len(chat_history) and chat_history[i + 1].get('role') == 'assistant':
                        assistant_response = chat_history[i + 1].get('content', '')
                    
                    # Get emotion analysis if available
                    emotion = message.get('emotion', 'neutral')
                    sentiment = message.get('sentiment', 'neutral')
                    
                    # Create searchable text
                    searchable_text = f"{user_text}\nEmotion: {emotion}\nSentiment: {sentiment}"
                    
                    # Create unique ID
                    doc_id = f"{session_id}_msg_{i}"
                    
                    # Add to vector database
                    self.conversations_collection.add(
                        documents=[searchable_text],
                        metadatas=[{
                            'user_id': user_id,
                            'session_id': session_id,
                            'message_index': i,
                            'timestamp': message.get('timestamp', ''),
                            'emotion': emotion,
                            'sentiment': sentiment,
                            'assistant_response': assistant_response[:500]  # Truncate
                        }],
                        ids=[doc_id]
                    )
            
            # Index session summary insights
            if 'summary' in session_data:
                summary = session_data['summary']
                insight_text = f"Session Summary: {json.dumps(summary)}"
                
                self.insights_collection.add(
                    documents=[insight_text],
                    metadatas=[{
                        'user_id': user_id,
                        'session_id': session_id,
                        'timestamp': session_data.get('saved_at', ''),
                        'type': 'session_summary'
                    }],
                    ids=[f"{session_id}_summary"]
                )
            
        except Exception as e:
            logger.warning(f"Failed to index session conversations: {e}")
    
    def find_similar_conversations(self, 
                                   user_id: str,
                                   query_text: str,
                                   n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar past conversations using semantic search
        """
        try:
            results = self.conversations_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"user_id": user_id}
            )
            
            similar_convos = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    
                    similar_convos.append({
                        'text': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'session_id': metadata.get('session_id', '')
                    })
            
            return similar_convos
            
        except Exception as e:
            logger.error(f"Failed to search similar conversations: {e}")
            return []
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from JSON file
        """
        try:
            session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
            
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def get_user_sessions(self, 
                          user_id: str,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user
        """
        try:
            user_sessions = []
            
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.sessions_dir, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session = json.load(f)
                        
                        if session.get('user_id') == user_id:
                            user_sessions.append(session)
            
            # Sort by date (most recent first)
            user_sessions.sort(
                key=lambda x: x.get('saved_at', ''),
                reverse=True
            )
            
            return user_sessions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def get_conversation_context(self,
                                 user_id: str,
                                 current_message: str,
                                 n_similar: int = 3) -> Dict[str, Any]:
        """
        Get relevant context from past conversations for current message
        """
        try:
            # Find similar past conversations
            similar = self.find_similar_conversations(user_id, current_message, n_similar)
            
            # Get recent sessions
            recent_sessions = self.get_user_sessions(user_id, limit=3)
            
            context = {
                'similar_conversations': similar,
                'recent_sessions': recent_sessions,
                'context_summary': self._generate_context_summary(similar, recent_sessions)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {'similar_conversations': [], 'recent_sessions': [], 'context_summary': ''}
    
    def _serialize_datetime(self, data: Any) -> Any:
        """
        Recursively convert datetime objects to ISO format strings
        """
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {key: self._serialize_datetime(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_datetime(item) for item in data]
        else:
            return data
    
    def _generate_context_summary(self,
                                   similar_convos: List[Dict],
                                   recent_sessions: List[Dict]) -> str:
        """
        Generate a summary of relevant context
        """
        summary_parts = []
        
        if similar_convos:
            summary_parts.append(f"Found {len(similar_convos)} similar past conversations:")
            for conv in similar_convos[:2]:  # Top 2
                emotion = conv.get('metadata', {}).get('emotion', 'unknown')
                summary_parts.append(f"- Similar topic discussed with {emotion} emotion")
        
        if recent_sessions:
            summary_parts.append(f"\nRecent activity: {len(recent_sessions)} sessions")
            latest = recent_sessions[0]
            summary_parts.append(f"Last session: {latest.get('saved_at', 'unknown')}")
        
        return '\n'.join(summary_parts)
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a user (GDPR compliance)
        """
        try:
            # Delete JSON files
            deleted_count = 0
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.sessions_dir, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session = json.load(f)
                        
                        if session.get('user_id') == user_id:
                            os.remove(filepath)
                            deleted_count += 1
            
            # Delete from vector database
            # Note: ChromaDB doesn't support batch delete by metadata yet
            # So we need to query and delete individually
            self.conversations_collection.delete(
                where={"user_id": user_id}
            )
            
            self.insights_collection.delete(
                where={"user_id": user_id}
            )
            
            logger.info(f"✅ Deleted {deleted_count} sessions for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize
    vsm = VectorSessionManager()
    
    # Test data
    test_session = {
        'session_id': 'test_session_001',
        'start_time': datetime.now().isoformat(),
        'chat_history': [
            {
                'role': 'user',
                'content': 'I am feeling really anxious about my job interview tomorrow',
                'emotion': 'anxious',
                'sentiment': 'negative',
                'timestamp': datetime.now().isoformat()
            },
            {
                'role': 'assistant',
                'content': 'I understand job interviews can feel overwhelming. Let\'s work through this anxiety together.',
                'timestamp': datetime.now().isoformat()
            }
        ]
    }
    
    # Save session
    vsm.save_session('test_user_123', 'test_session_001', test_session)
    
    # Search for similar
    similar = vsm.find_similar_conversations('test_user_123', 'I am worried about my presentation')
    print(f"\nFound {len(similar)} similar conversations")
    
    for conv in similar:
        print(f"\nSimilarity: {conv['similarity_score']:.2f}")
        print(f"Text: {conv['text'][:100]}...")
