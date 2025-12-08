"""
Agentic Mental Health Therapy System with User-Controlled Persistence
Implements true agent behavior with encrypted local storage and user privacy controls
"""

import logging
import json
import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import requests
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserGoal:
    """Represents a therapeutic goal set by the user"""
    id: str
    title: str
    description: str
    target_date: str
    progress_score: float  # 0-100
    status: str  # active, completed, paused
    created_date: str
    techniques_used: List[str]
    milestones: List[Dict[str, Any]]

@dataclass
class SessionSummary:
    """Summary of a therapy session"""
    session_id: str
    date: str
    duration_minutes: int
    dominant_emotions: List[str]
    topics_discussed: List[str]
    techniques_used: List[str]
    effectiveness_rating: float  # 1-10
    crisis_indicators: List[str]
    key_insights: List[str]
    homework_assigned: List[str]

@dataclass
class UserProfile:
    """Comprehensive user profile for personalization"""
    user_id: str
    preferred_name: str
    personality_traits: Dict[str, float]
    trigger_patterns: List[str]
    effective_techniques: Dict[str, float]
    communication_style: str
    risk_factors: List[str]
    progress_metrics: Dict[str, float]
    last_activity: str

class EncryptionManager:
    """Handles local encryption for user data"""
    
    def __init__(self, user_password: str = None):
        self.key = None
        if user_password:
            self.set_password(user_password)
    
    def set_password(self, password: str):
        """Generate encryption key from user password"""
        password_bytes = password.encode('utf-8')
        salt = b'therapy_ai_salt_2024'  # In production, use random salt per user
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.key = key
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.key:
            raise ValueError("Encryption key not set")
        f = Fernet(self.key)
        encrypted_data = f.encrypt(data.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.key:
            raise ValueError("Encryption key not set")
        f = Fernet(self.key)
        decoded_data = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted_data = f.decrypt(decoded_data)
        return decrypted_data.decode('utf-8')

class UserMemoryManager:
    """Manages persistent user memory with encryption"""
    
    def __init__(self, db_path: str = "user_memory.db"):
        self.db_path = db_path
        self.encryption_manager = None
        self.current_user_id = None
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with encrypted tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                encrypted_profile TEXT NOT NULL,
                retention_days INTEGER DEFAULT 30,
                created_date TEXT NOT NULL,
                last_access TEXT NOT NULL
            )
        ''')
        
        # Session summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                encrypted_summary TEXT NOT NULL,
                session_date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')
        
        # Goals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_goals (
                goal_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                encrypted_goal TEXT NOT NULL,
                created_date TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')
        
        # Privacy consents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS privacy_consents (
                user_id TEXT PRIMARY KEY,
                memory_consent BOOLEAN DEFAULT FALSE,
                retention_days INTEGER DEFAULT 7,
                consent_date TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def authenticate_user(self, user_id: str, password: str) -> bool:
        """Authenticate user and set up encryption"""
        try:
            self.encryption_manager = EncryptionManager(password)
            self.current_user_id = user_id
            
            # Test decryption with existing data
            profile = self.get_user_profile(user_id)
            return True
        except Exception as e:
            logger.warning(f"Authentication failed for user {user_id}: {e}")
            return False
    
    def create_user(self, user_id: str, password: str, preferred_name: str) -> bool:
        """Create new user with encrypted profile"""
        try:
            self.encryption_manager = EncryptionManager(password)
            self.current_user_id = user_id
            
            # Create initial user profile
            profile = UserProfile(
                user_id=user_id,
                preferred_name=preferred_name,
                personality_traits={},
                trigger_patterns=[],
                effective_techniques={},
                communication_style="supportive",
                risk_factors=[],
                progress_metrics={},
                last_activity=datetime.now().isoformat()
            )
            
            self.save_user_profile(profile)
            return True
        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}")
            return False
    
    def save_user_profile(self, profile: UserProfile):
        """Save encrypted user profile"""
        if not self.encryption_manager:
            raise ValueError("User not authenticated")
        
        profile_json = json.dumps(asdict(profile))
        encrypted_profile = self.encryption_manager.encrypt_data(profile_json)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles 
            (user_id, encrypted_profile, created_date, last_access)
            VALUES (?, ?, ?, ?)
        ''', (profile.user_id, encrypted_profile, datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve and decrypt user profile"""
        if not self.encryption_manager:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT encrypted_profile FROM user_profiles WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                decrypted_data = self.encryption_manager.decrypt_data(result[0])
                profile_dict = json.loads(decrypted_data)
                return UserProfile(**profile_dict)
            except Exception as e:
                logger.error(f"Failed to decrypt profile for {user_id}: {e}")
                return None
        return None
    
    def save_session_summary(self, summary: SessionSummary):
        """Save encrypted session summary"""
        if not self.encryption_manager:
            raise ValueError("User not authenticated")
        
        summary_json = json.dumps(asdict(summary))
        encrypted_summary = self.encryption_manager.encrypt_data(summary_json)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO session_summaries 
            (session_id, user_id, encrypted_summary, session_date)
            VALUES (?, ?, ?, ?)
        ''', (summary.session_id, self.current_user_id, encrypted_summary, summary.date))
        
        conn.commit()
        conn.close()
    
    def get_recent_sessions(self, user_id: str, limit: int = 5) -> List[SessionSummary]:
        """Get recent session summaries for context"""
        if not self.encryption_manager:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT encrypted_summary FROM session_summaries 
            WHERE user_id = ? 
            ORDER BY session_date DESC LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        sessions = []
        for result in results:
            try:
                decrypted_data = self.encryption_manager.decrypt_data(result[0])
                session_dict = json.loads(decrypted_data)
                sessions.append(SessionSummary(**session_dict))
            except Exception as e:
                logger.warning(f"Failed to decrypt session: {e}")
        
        return sessions
    
    def save_user_goal(self, goal: UserGoal):
        """Save encrypted user goal"""
        if not self.encryption_manager:
            raise ValueError("User not authenticated")
        
        goal_json = json.dumps(asdict(goal))
        encrypted_goal = self.encryption_manager.encrypt_data(goal_json)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_goals 
            (goal_id, user_id, encrypted_goal, created_date, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (goal.id, self.current_user_id, encrypted_goal, goal.created_date, goal.status))
        
        conn.commit()
        conn.close()
    
    def get_active_goals(self, user_id: str) -> List[UserGoal]:
        """Get active goals for user"""
        if not self.encryption_manager:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT encrypted_goal FROM user_goals 
            WHERE user_id = ? AND status = 'active'
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        goals = []
        for result in results:
            try:
                decrypted_data = self.encryption_manager.decrypt_data(result[0])
                goal_dict = json.loads(decrypted_data)
                goals.append(UserGoal(**goal_dict))
            except Exception as e:
                logger.warning(f"Failed to decrypt goal: {e}")
        
        return goals
    
    def delete_user_data(self, user_id: str):
        """Completely delete all user data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_profiles WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM session_summaries WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM user_goals WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM privacy_consents WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"All data deleted for user {user_id}")
    
    def cleanup_expired_data(self):
        """Clean up data past retention period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get users with retention periods
        cursor.execute('''
            SELECT p.user_id, c.retention_days 
            FROM user_profiles p 
            JOIN privacy_consents c ON p.user_id = c.user_id
        ''')
        
        for user_id, retention_days in cursor.fetchall():
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_str = cutoff_date.isoformat()
            
            # Delete old sessions
            cursor.execute('''
                DELETE FROM session_summaries 
                WHERE user_id = ? AND session_date < ?
            ''', (user_id, cutoff_str))
        
        conn.commit()
        conn.close()

class GoalTracker:
    """Manages therapeutic goals and progress tracking"""
    
    def __init__(self, memory_manager: UserMemoryManager):
        self.memory_manager = memory_manager
    
    def create_goal(self, user_id: str, title: str, description: str, target_days: int = 30) -> UserGoal:
        """Create a new therapeutic goal"""
        goal_id = f"goal_{secrets.token_hex(8)}"
        target_date = (datetime.now() + timedelta(days=target_days)).isoformat()
        
        goal = UserGoal(
            id=goal_id,
            title=title,
            description=description,
            target_date=target_date,
            progress_score=0.0,
            status="active",
            created_date=datetime.now().isoformat(),
            techniques_used=[],
            milestones=[]
        )
        
        self.memory_manager.save_user_goal(goal)
        return goal
    
    def update_progress(self, goal_id: str, progress_increment: float, technique_used: str = None):
        """Update goal progress based on session outcomes"""
        goals = self.memory_manager.get_active_goals(self.memory_manager.current_user_id)
        
        for goal in goals:
            if goal.id == goal_id:
                goal.progress_score = min(100.0, goal.progress_score + progress_increment)
                
                if technique_used and technique_used not in goal.techniques_used:
                    goal.techniques_used.append(technique_used)
                
                # Add milestone if significant progress
                if goal.progress_score >= 25 and len(goal.milestones) == 0:
                    goal.milestones.append({
                        "milestone": "25% Progress",
                        "date": datetime.now().isoformat(),
                        "note": "Great start on your therapeutic journey!"
                    })
                elif goal.progress_score >= 50 and len(goal.milestones) == 1:
                    goal.milestones.append({
                        "milestone": "50% Progress", 
                        "date": datetime.now().isoformat(),
                        "note": "Halfway there! Keep up the good work."
                    })
                elif goal.progress_score >= 75 and len(goal.milestones) == 2:
                    goal.milestones.append({
                        "milestone": "75% Progress",
                        "date": datetime.now().isoformat(), 
                        "note": "Excellent progress! You're almost there."
                    })
                
                if goal.progress_score >= 100:
                    goal.status = "completed"
                    goal.milestones.append({
                        "milestone": "Goal Completed",
                        "date": datetime.now().isoformat(),
                        "note": "Congratulations! You've achieved your goal."
                    })
                
                self.memory_manager.save_user_goal(goal)
                break

class AdaptiveLearningEngine:
    """Learns from user interactions to improve recommendations"""
    
    def __init__(self, memory_manager: UserMemoryManager):
        self.memory_manager = memory_manager
    
    def analyze_technique_effectiveness(self, user_id: str) -> Dict[str, float]:
        """Analyze which techniques work best for this user"""
        sessions = self.memory_manager.get_recent_sessions(user_id, limit=20)
        technique_scores = {}
        
        for session in sessions:
            for technique in session.techniques_used:
                if technique not in technique_scores:
                    technique_scores[technique] = []
                technique_scores[technique].append(session.effectiveness_rating)
        
        # Calculate average effectiveness for each technique
        avg_scores = {}
        for technique, scores in technique_scores.items():
            avg_scores[technique] = sum(scores) / len(scores)
        
        return avg_scores
    
    def predict_optimal_techniques(self, user_id: str, current_emotion: str) -> List[str]:
        """Predict which techniques will work best for current state"""
        profile = self.memory_manager.get_user_profile(user_id)
        if not profile:
            return ["supportive_counseling"]
        
        effectiveness = self.analyze_technique_effectiveness(user_id)
        
        # Sort techniques by effectiveness
        sorted_techniques = sorted(effectiveness.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 3 techniques
        return [technique for technique, score in sorted_techniques[:3]]
    
    def update_user_model(self, user_id: str, session_summary: SessionSummary):
        """Update user model based on session outcomes"""
        profile = self.memory_manager.get_user_profile(user_id)
        if not profile:
            return
        
        # Update effective techniques
        for technique in session_summary.techniques_used:
            if technique not in profile.effective_techniques:
                profile.effective_techniques[technique] = 0.0
            
            # Weight by effectiveness rating
            current_score = profile.effective_techniques[technique]
            new_score = (current_score + session_summary.effectiveness_rating) / 2
            profile.effective_techniques[technique] = new_score
        
        # Update trigger patterns
        if session_summary.crisis_indicators:
            for indicator in session_summary.crisis_indicators:
                if indicator not in profile.trigger_patterns:
                    profile.trigger_patterns.append(indicator)
        
        # Update progress metrics
        profile.progress_metrics[datetime.now().strftime("%Y-%m")] = session_summary.effectiveness_rating
        profile.last_activity = datetime.now().isoformat()
        
        self.memory_manager.save_user_profile(profile)

class ProactiveInterventionEngine:
    """Handles proactive interventions and check-ins"""
    
    def __init__(self, memory_manager: UserMemoryManager, goal_tracker: GoalTracker):
        self.memory_manager = memory_manager
        self.goal_tracker = goal_tracker
    
    def should_initiate_checkin(self, user_id: str) -> bool:
        """Determine if a proactive check-in is needed"""
        profile = self.memory_manager.get_user_profile(user_id)
        if not profile:
            return False
        
        last_activity = datetime.fromisoformat(profile.last_activity)
        days_since_activity = (datetime.now() - last_activity).days
        
        # Check-in logic
        if days_since_activity >= 3:  # No activity for 3+ days
            return True
        
        recent_sessions = self.memory_manager.get_recent_sessions(user_id, limit=3)
        if recent_sessions:
            avg_effectiveness = sum(s.effectiveness_rating for s in recent_sessions) / len(recent_sessions)
            if avg_effectiveness < 5:  # Low effectiveness trend
                return True
        
        return False
    
    def generate_checkin_message(self, user_id: str) -> str:
        """Generate personalized check-in message"""
        profile = self.memory_manager.get_user_profile(user_id)
        goals = self.memory_manager.get_active_goals(user_id)
        
        messages = [
            f"Hi {profile.preferred_name}, I wanted to check in with you. How are you feeling today?",
            f"Hello {profile.preferred_name}, it's been a few days. What's on your mind?",
            f"Hey {profile.preferred_name}, I'm thinking of you. How has your week been?"
        ]
        
        if goals:
            goal_message = f"I noticed you're working on '{goals[0].title}'. How is that going?"
            messages.append(goal_message)
        
        return random.choice(messages)
    
    def detect_intervention_opportunity(self, user_message: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Detect opportunities for specific interventions"""
        user_id = self.memory_manager.current_user_id
        if not user_id:
            return None
        
        profile = self.memory_manager.get_user_profile(user_id)
        if not profile:
            return None
        
        # Check for known triggers
        message_lower = user_message.lower()
        for trigger in profile.trigger_patterns:
            if trigger.lower() in message_lower:
                effective_techniques = profile.effective_techniques
                if effective_techniques:
                    best_technique = max(effective_techniques.items(), key=lambda x: x[1])[0]
                    return f"I notice this situation seems familiar. Last time, {best_technique} helped you. Would you like to try that again?"
        
        return None

class AgenticTherapySystem:
    """Main agentic therapy system integrating all components"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.memory_manager = UserMemoryManager()
        self.goal_tracker = GoalTracker(self.memory_manager)
        self.learning_engine = AdaptiveLearningEngine(self.memory_manager)
        self.intervention_engine = ProactiveInterventionEngine(self.memory_manager, self.goal_tracker)
        
        # Initialize LLM components (from your existing code)
        self.ollama_url = self.config.get("llm", {}).get("ollama_url", "http://localhost:11434")
        self.groq_api_key = self.config.get("llm", {}).get("groq_api_key")
        self.groq_model = "llama-3.1-8b-instant"
        
        # Privacy settings
        self.privacy_mode = True  # Start in privacy mode by default
        self.current_user_id = None
        
        logger.info("Agentic Therapy System initialized")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return {}
    
    def request_privacy_consent(self) -> dict:
        """Request user consent for data persistence"""
        consent_request = {
            "message": """I can provide better, personalized support if I remember our conversations and track your progress over time. This would allow me to:

• Remember what techniques work best for you
• Track your therapeutic goals and progress  
• Provide continuity between our sessions
• Offer personalized check-ins and suggestions

All data would be:
• Stored only on your device (never uploaded)
• Encrypted with your password
• Deletable by you at any time
• Kept only as long as you choose

Would you like me to remember our conversations, or prefer to keep each session private and separate?""",
            "options": [
                {"id": "remember", "text": "Yes, remember our conversations", "retention_days": [7, 30, 90, 365]},
                {"id": "private", "text": "No, keep sessions private", "retention_days": 0}
            ]
        }
        return consent_request
    
    def handle_privacy_consent(self, user_choice: dict) -> dict:
        """Handle user's privacy consent decision"""
        if user_choice.get("choice") == "remember":
            retention_days = user_choice.get("retention_days", 30)
            user_id = user_choice.get("user_id") or f"user_{secrets.token_hex(8)}"
            password = user_choice.get("password") or secrets.token_urlsafe(16)
            preferred_name = user_choice.get("preferred_name", "there")
            
            # Create user with encrypted storage
            success = self.memory_manager.create_user(user_id, password, preferred_name)
            
            if success:
                self.privacy_mode = False
                self.current_user_id = user_id
                
                return {
                    "status": "success",
                    "message": f"Perfect! I'll remember our conversations and keep them secure. Your data will be automatically deleted after {retention_days} days unless you change this setting.",
                    "user_id": user_id,
                    "password": password,  # Show once for user to save
                    "retention_days": retention_days
                }
            else:
                return {
                    "status": "error",
                    "message": "Sorry, I couldn't set up secure storage. Let's continue with private sessions."
                }
        else:
            self.privacy_mode = True
            return {
                "status": "success", 
                "message": "Understood. I'll keep our sessions private and won't remember between conversations."
            }
    
    def authenticate_returning_user(self, user_id: str, password: str) -> dict:
        """Authenticate returning user"""
        success = self.memory_manager.authenticate_user(user_id, password)
        
        if success:
            self.privacy_mode = False
            self.current_user_id = user_id
            profile = self.memory_manager.get_user_profile(user_id)
            
            # Check if profile was successfully retrieved
            if not profile:
                return {
                    "status": "error",
                    "message": "Failed to load user profile. Please try again or start a new session."
                }
            
            # Check if proactive check-in needed
            checkin_needed = self.intervention_engine.should_initiate_checkin(user_id)
            
            response = {
                "status": "success",
                "message": f"Welcome back, {profile.preferred_name}! I remember our previous conversations.",
                "proactive_checkin": checkin_needed
            }
            
            if checkin_needed:
                response["checkin_message"] = self.intervention_engine.generate_checkin_message(user_id)
            
            return response
        else:
            return {
                "status": "error",
                "message": "Authentication failed. Please check your credentials or start a new private session."
            }
    
    def generate_agentic_response(self, user_message: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response with full agentic capabilities"""
        
        # Check for proactive intervention opportunities
        intervention = None
        if not self.privacy_mode and self.current_user_id:
            intervention = self.intervention_engine.detect_intervention_opportunity(user_message, analysis)
        
        # Get personalized context
        context = self.build_personalized_context(user_message, analysis)
        
        # Generate response using LLM with enhanced context
        response = self.generate_llm_response(user_message, analysis, context)
        
        # Update user model if not in privacy mode
        if not self.privacy_mode and self.current_user_id:
            self.update_user_learning(user_message, analysis, response)
        
        # Add agentic elements to response
        if intervention:
            response["proactive_suggestion"] = intervention
        
        # Add goal-related suggestions
        if not self.privacy_mode and self.current_user_id:
            goals = self.memory_manager.get_active_goals(self.current_user_id)
            if goals:
                response["goal_progress"] = self.format_goal_progress(goals)
        
        return response
    
    def build_personalized_context(self, user_message: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build personalized context for response generation"""
        context = {
            "user_message": user_message,
            "analysis": analysis,
            "privacy_mode": self.privacy_mode
        }
        
        if not self.privacy_mode and self.current_user_id:
            profile = self.memory_manager.get_user_profile(self.current_user_id)
            recent_sessions = self.memory_manager.get_recent_sessions(self.current_user_id, limit=3)
            effective_techniques = self.learning_engine.predict_optimal_techniques(
                self.current_user_id, 
                analysis.get('dominant_emotion', 'neutral')
            )
            
            context.update({
                "user_profile": profile,
                "recent_sessions": recent_sessions,
                "effective_techniques": effective_techniques,
                "personalization_available": True
            })
        
        return context
    
    def generate_llm_response(self, user_message: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using LLM with enhanced context"""
        # Build enhanced prompt with personalization
        prompt = self.build_enhanced_prompt(user_message, analysis, context)
        
        # Try LLM providers (your existing logic)
        response = None
        
        # Try Groq first for personalized responses
        if self.groq_api_key and not self.privacy_mode:
            response = self.try_groq_with_context(prompt, context)
        
        # Fallback to rule-based with personalization
        if not response:
            response = self.generate_personalized_fallback(user_message, analysis, context)
        
        return response
    
    def build_enhanced_prompt(self, user_message: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build enhanced prompt with personalization and conversation history"""
        base_prompt = f"""You are a compassionate mental health therapist. 

User's current message: "{user_message}"
Emotion detected: {analysis.get('dominant_emotion', 'neutral')}
Risk level: {analysis.get('risk_level', 'LOW')}"""
        
        if context.get("personalization_available") and not self.privacy_mode:
            profile = context["user_profile"]
            effective_techniques = context["effective_techniques"]
            recent_sessions = context.get("recent_sessions", [])
            
            personalization = f"""

PERSONALIZATION CONTEXT:
- User prefers to be called: {profile.preferred_name}
- Most effective techniques for this user: {', '.join(effective_techniques)}
- Communication style: {profile.communication_style}
- Previous successful approaches: {list(profile.effective_techniques.keys())[:3]}"""

            # Add recent conversation history for context
            if recent_sessions:
                personalization += "\n\nRECENT CONVERSATION HISTORY:"
                for i, session in enumerate(recent_sessions[-2:], 1):  # Last 2 sessions
                    if session.key_insights:
                        personalization += f"\nSession {i}: {'; '.join(session.key_insights)}"
                
                personalization += "\n\nImportant: Reference previous conversations when relevant. If the user asks about something you discussed before, acknowledge that you remember and build upon that conversation."
            
            base_prompt += personalization
        
        base_prompt += """

CRITICAL SAFETY NOTE: If user mentions violence, self-harm, or hurting others, immediately provide crisis resources and encourage professional help.

Provide an empathetic, therapeutic response that:
1. Acknowledges their feelings with empathy
2. References previous conversations if relevant (for agentic mode)
3. Uses techniques that work for this user (if known)
4. Offers specific, actionable suggestions
5. Maintains therapeutic boundaries
6. Provides crisis resources if needed
"""
        
        return base_prompt
    
    def try_groq_with_context(self, prompt: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try Groq API with enhanced context and better error handling"""
        try:
            logger.info("Making Groq API request...")
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context["user_message"]}
            ]
            
            payload = {
                "messages": messages,
                "model": self.groq_model,
                "temperature": 0.7,
                "max_tokens": 300
            }
            
            logger.info(f"Groq request: {self.groq_model}, {len(messages)} messages")
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            )
            
            logger.info(f"Groq response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                if content:
                    logger.info(f"SUCCESS: Groq generated {len(content)} characters")
                    return {
                        'content': content,
                        'provider': 'groq',
                        'model': self.groq_model,
                        'personalized': not self.privacy_mode,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    logger.warning("Groq returned empty content")
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            logger.error("Groq API timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected Groq error: {e}")
        
        return None
    
    def _try_ollama_direct(self, user_message: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try Ollama directly with enhanced context"""
        try:
            logger.info("Making Ollama API request...")
            
            # Build context for Ollama
            therapeutic_context = self._build_therapeutic_context(user_message, analysis, context.get('recent_sessions', []))
            
            payload = {
                "model": "llama3.1:8b",
                "prompt": therapeutic_context,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 300
                }
            }
            
            logger.info(f"Ollama request: {payload['model']}")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60  # Increased timeout for Ollama
            )
            
            logger.info(f"Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                if content:
                    logger.info(f"SUCCESS: Ollama generated {len(content)} characters")
                    return {
                        'content': content,
                        'provider': 'ollama',
                        'model': payload['model'],
                        'personalized': not self.privacy_mode,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    logger.warning("Ollama returned empty content")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("Ollama API timeout (60s)")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
        
        return None
    
    def _build_therapeutic_context(self, user_message: str, analysis: Dict[str, Any], recent_sessions: List = None) -> str:
        """Build therapeutic context for Ollama with personalization"""
        
        emotion = analysis.get('dominant_emotion', 'neutral')
        sentiment = analysis.get('sentiment', 'neutral')
        risk_level = analysis.get('risk_level', 'LOW')
        topics = analysis.get('mental_health_topics', [])
        techniques = analysis.get('suggested_techniques', [])
        
        context = f"""You are a compassionate, professional mental health therapist. Respond to the user with empathy and therapeutic techniques.

Analysis of user's message:
- Emotion: {emotion}
- Sentiment: {sentiment}  
- Risk level: {risk_level}
- Topics: {', '.join([t[0] for t in topics[:3]])}
- Suggested techniques: {', '.join(techniques)}"""

        # Add personalization if available
        if not self.privacy_mode and self.current_user_id:
            try:
                profile = self.memory_manager.get_user_profile(self.current_user_id)
                if profile:
                    effective_techniques = list(profile.effective_techniques.keys())[:3]
                    context += f"""

PERSONALIZATION CONTEXT:
- User prefers to be called: {profile.preferred_name}
- Most effective techniques for this user: {', '.join(effective_techniques)}
- Communication style: {profile.communication_style}

Tailor your response to this specific user's preferences and what has worked for them before."""
            except Exception as e:
                logger.error(f"Failed to add personalization context: {e}")

        context += f"""

User's message: "{user_message}"

Provide a therapeutic response that:
1. Acknowledges their feelings with empathy
2. Uses techniques that work for this user (if known)
3. Offers specific, actionable suggestions
4. Maintains therapeutic boundaries
5. Keep response concise (2-3 sentences)

Therapeutic response:"""
        
        return context
    
    def generate_personalized_fallback(self, user_message: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized fallback response"""
        
        if context.get("personalization_available"):
            profile = context["user_profile"]
            effective_techniques = context["effective_techniques"]
            
            # Use known effective techniques
            if effective_techniques:
                technique = effective_techniques[0]
                response_content = f"Hi {profile.preferred_name}, I remember that {technique} has been helpful for you before. "
            else:
                response_content = f"Hi {profile.preferred_name}, let's work through this together. "
        else:
            response_content = "I hear you, and I'm here to help. "
        
        # Add situation-specific guidance
        emotion = analysis.get('dominant_emotion', 'neutral')
        if emotion == 'anxiety':
            response_content += "When anxiety feels overwhelming, try the 4-7-8 breathing technique: breathe in for 4, hold for 7, exhale for 8. What feels most manageable right now?"
        elif emotion == 'sadness':
            response_content += "It's okay to feel sad. Small steps can help: reach out to someone you trust, do one self-care activity, or simply rest without guilt. What sounds possible today?"
        else:
            response_content += "Let's focus on what you can control today. What's one small step that might help you feel a bit better?"
        
        return {
            'content': response_content,
            'provider': 'personalized_fallback',
            'personalized': not self.privacy_mode,
            'timestamp': datetime.now().isoformat()
        }
    
    def update_user_learning(self, user_message: str, analysis: Dict[str, Any], response: Dict[str, Any]):
        """Update user model based on interaction"""
        if self.privacy_mode or not self.current_user_id:
            return
        
        # Create session summary (simplified - you'd get effectiveness rating from user feedback)
        session_summary = SessionSummary(
            session_id=f"session_{secrets.token_hex(8)}",
            date=datetime.now().isoformat(),
            duration_minutes=5,  # Estimated
            dominant_emotions=[analysis.get('dominant_emotion', 'neutral')],
            topics_discussed=analysis.get('mental_health_topics', []),
            techniques_used=['supportive_counseling'],  # Would extract from response
            effectiveness_rating=7.0,  # Would get from user feedback
            crisis_indicators=analysis.get('crisis_indicators', []),
            key_insights=[],
            homework_assigned=[]
        )
        
        # Save session and update learning
        self.memory_manager.save_session_summary(session_summary)
        self.learning_engine.update_user_model(self.current_user_id, session_summary)
    
    def format_goal_progress(self, goals: List[UserGoal]) -> Dict[str, Any]:
        """Format goal progress for display"""
        if not goals:
            return {}
        
        active_goal = goals[0]  # Show primary goal
        return {
            "goal_title": active_goal.title,
            "progress_percentage": active_goal.progress_score,
            "recent_milestone": active_goal.milestones[-1] if active_goal.milestones else None,
            "encouragement": self.generate_progress_encouragement(active_goal)
        }
    
    def generate_progress_encouragement(self, goal: UserGoal) -> str:
        """Generate encouraging message based on goal progress"""
        progress = goal.progress_score
        
        if progress < 25:
            return "Every step counts! You're building important foundations."
        elif progress < 50:
            return "Great progress! You're developing real momentum."
        elif progress < 75:
            return "Excellent work! You're more than halfway to your goal."
        else:
            return "Outstanding progress! You're so close to achieving your goal."
    
    def create_user_goal(self, title: str, description: str, target_days: int = 30) -> Dict[str, Any]:
        """Create a new therapeutic goal for the user"""
        if self.privacy_mode or not self.current_user_id:
            return {"error": "Goal tracking requires persistent memory consent"}
        
        goal = self.goal_tracker.create_goal(self.current_user_id, title, description, target_days)
        
        return {
            "status": "success",
            "goal": asdict(goal),
            "message": f"Great! I've created your goal: '{title}'. I'll help you track progress and celebrate milestones along the way."
        }
    
    def get_user_dashboard(self) -> Dict[str, Any]:
        """Get user dashboard with goals, progress, and insights"""
        if self.privacy_mode or not self.current_user_id:
            return {"message": "Dashboard requires persistent memory consent"}
        
        profile = self.memory_manager.get_user_profile(self.current_user_id)
        goals = self.memory_manager.get_active_goals(self.current_user_id)
        recent_sessions = self.memory_manager.get_recent_sessions(self.current_user_id, limit=7)
        
        # Calculate insights
        effectiveness_trend = [s.effectiveness_rating for s in recent_sessions]
        avg_effectiveness = sum(effectiveness_trend) / len(effectiveness_trend) if effectiveness_trend else 0
        
        return {
            "user_profile": {
                "preferred_name": profile.preferred_name,
                "last_activity": profile.last_activity,
                "total_sessions": len(recent_sessions)
            },
            "active_goals": [asdict(goal) for goal in goals],
            "insights": {
                "average_session_effectiveness": avg_effectiveness,
                "most_effective_techniques": list(profile.effective_techniques.keys())[:3],
                "recent_progress": "positive" if avg_effectiveness >= 6 else "needs_attention"
            },
            "suggestions": self.generate_dashboard_suggestions(profile, goals, avg_effectiveness)
        }
    
    def generate_dashboard_suggestions(self, profile: UserProfile, goals: List[UserGoal], avg_effectiveness: float) -> List[str]:
        """Generate personalized suggestions for dashboard"""
        suggestions = []
        
        if avg_effectiveness < 5:
            suggestions.append("Consider trying a different therapeutic approach - let's explore what might work better for you.")
        
        if goals and goals[0].progress_score < 20:
            suggestions.append("Let's break down your goal into smaller, more manageable steps.")
        
        if not goals:
            suggestions.append("Setting a therapeutic goal could help focus our sessions. Would you like to create one?")
        
        return suggestions
    
    def delete_all_user_data(self) -> Dict[str, Any]:
        """Delete all user data per user request"""
        if self.privacy_mode or not self.current_user_id:
            return {"message": "No data to delete in privacy mode"}
        
        user_id = self.current_user_id
        self.memory_manager.delete_user_data(user_id)
        
        # Reset to privacy mode
        self.privacy_mode = True
        self.current_user_id = None
        
        return {
            "status": "success",
            "message": "All your data has been permanently deleted. Future sessions will be private unless you choose to enable memory again."
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    therapy_system = AgenticTherapySystem()
    
    # Test privacy consent flow
    print("=== Privacy Consent Test ===")
    consent_request = therapy_system.request_privacy_consent()
    print(consent_request["message"])
    
    # Simulate user choosing to remember
    user_choice = {
        "choice": "remember",
        "retention_days": 30,
        "preferred_name": "Alex",
        "user_id": "test_user_123",
        "password": "secure_password_123"
    }
    
    consent_result = therapy_system.handle_privacy_consent(user_choice)
    print(f"\nConsent result: {consent_result['message']}")
    
    # Test agentic response
    print("\n=== Agentic Response Test ===")
    test_message = "I've been feeling anxious about work lately"
    test_analysis = {
        'dominant_emotion': 'anxiety',
        'sentiment': 'negative',
        'risk_level': 'LOW',
        'mental_health_topics': [('anxiety', 0.8)],
        'suggested_techniques': ['mindfulness', 'breathing']
    }
    
    response = therapy_system.generate_agentic_response(test_message, test_analysis)
    print(f"Response: {response['content']}")
    
    # Test goal creation
    print("\n=== Goal Creation Test ===")
    goal_result = therapy_system.create_user_goal(
        "Manage work anxiety",
        "Learn techniques to stay calm during stressful work situations",
        30
    )
    print(f"Goal created: {goal_result['message']}")
    
    # Test dashboard
    print("\n=== Dashboard Test ===")
    dashboard = therapy_system.get_user_dashboard()
    print(f"Dashboard insights: {dashboard['insights']}")
    
    print("\n=== System Test Complete ===")