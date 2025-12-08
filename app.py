"""
Enhanced Flask Backend for Agentic AI Therapy System
Integrates user-controlled persistence, goal tracking, and adaptive learning
"""

from flask import Flask, request, jsonify, render_template, session, Response
from flask_cors import CORS
import uuid
import logging
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Any, Optional
import threading
import random

# Prevent transformers from importing TensorFlow/Flax, avoiding protobuf runtime_version errors
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
from dotenv import load_dotenv

# Import our modules
from text_analyzer import TextAnalyzer
from therapy_agent import TherapyAgent
from agentic_therapy_system import AgenticTherapySystem
from voice_agent import VoiceAgent
from video_agent import VideoAgent
from crisis_counselling_mode import CrisisCounsellingMode
from crisis_api import crisis_bp, init_crisis_api

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
CORS(app)

# Register blueprints
app.register_blueprint(crisis_bp)

# Global components
text_analyzer = None
therapy_agent = None
agentic_system = None
voice_agent = None
video_agent = None
crisis_counselor = None

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.cleanup_interval = timedelta(hours=24)
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'id': session_id,
            'created': datetime.now(),
            'last_activity': datetime.now(),
            'consent_given': False,
            'privacy_consent_requested': False,
            'agentic_mode': False,
            'user_id': None,
            'chat_history': [],
            'settings': {
                'crisis_sensitivity': 'medium',
                'analysis_depth': 'standard'
            },
            'stats': {
                'message_count': 0,
                'session_duration': 0
            }
        }
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        session_data = self.sessions.get(session_id)
        if session_data:
            session_data['last_activity'] = datetime.now()
        return session_data
    
    def update_session(self, session_id: str, data: Dict):
        if session_id in self.sessions:
            self.sessions[session_id].update(data)
            self.sessions[session_id]['last_activity'] = datetime.now()
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def cleanup_expired_sessions(self):
        cutoff = datetime.now() - self.cleanup_interval
        expired = [
            sid for sid, data in self.sessions.items()
            if data['last_activity'] < cutoff
        ]
        for sid in expired:
            self.delete_session(sid)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

session_manager = SessionManager()

def initialize_components():
    global text_analyzer, therapy_agent, agentic_system, voice_agent, video_agent, crisis_counselor

    logger.info("Initializing AI components...")

    try:
        text_analyzer = TextAnalyzer()
        logger.info("✓ Text analyzer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize text analyzer: {e}")
        text_analyzer = None

    try:
        therapy_agent = TherapyAgent()
        logger.info("✓ Therapy agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize therapy agent: {e}")
        therapy_agent = None

    try:
        agentic_system = AgenticTherapySystem()
        logger.info("✓ Agentic therapy system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize agentic therapy system: {e}")
        agentic_system = None

    try:
        voice_agent = VoiceAgent()
        logger.info("✓ Voice agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize voice agent: {e}")
        voice_agent = None

    try:
        video_agent = VideoAgent()
        logger.info("✓ Video agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize video agent: {e}")
        video_agent = None

    try:
        crisis_counselor = CrisisCounsellingMode()
        logger.info("✓ Crisis counselor initialized")
        # Initialize crisis API with components
        init_crisis_api(crisis_counselor, text_analyzer)
    except Exception as e:
        logger.error(f"Failed to initialize crisis counselor: {e}")
        crisis_counselor = None

    if text_analyzer is None or agentic_system is None:
        logger.warning("Some components failed to initialize - app will run with limited functionality")

_initialized = False
_init_lock = threading.Lock()

def ensure_initialized():
    global _initialized
    if not _initialized:
        with _init_lock:
            if not _initialized:
                initialize_components()
                _initialized = True

@app.route('/')
def index():
    ensure_initialized()
    return render_template('index.html')

@app.route('/health')
def health_check():
    ensure_initialized()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'text_analyzer': text_analyzer is not None,
            'therapy_agent': therapy_agent is not None,
            'agentic_system': agentic_system is not None,
            'voice_agent': voice_agent is not None,
            'video_agent': video_agent is not None,
            'crisis_counselor': crisis_counselor is not None
        }
    })

@app.route('/api/session/new', methods=['POST'])
def create_new_session():
    try:
        ensure_initialized()
        session_id = session_manager.create_session()
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session created successfully'
        })
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to create session'
        }), 500

@app.route('/api/privacy/consent/request', methods=['POST'])
def request_privacy_consent():
    """Request privacy consent from user for agentic features"""
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        # Get consent request from agentic system
        if agentic_system:
            consent_request = agentic_system.request_privacy_consent()
        else:
            consent_request = {
                "message": "Would you like me to remember our conversations to provide better support?",
                "options": [
                    {"id": "remember", "text": "Yes, remember our conversations", "retention_days": [7, 30, 90]},
                    {"id": "private", "text": "No, keep sessions private", "retention_days": 0}
                ]
            }
        
        session_manager.update_session(session_id, {
            'privacy_consent_requested': True
        })
        
        return jsonify({
            'success': True,
            'consent_request': consent_request
        })
        
    except Exception as e:
        logger.error(f"Error requesting privacy consent: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to request privacy consent'
        }), 500

@app.route('/api/privacy/consent/respond', methods=['POST'])
def handle_privacy_consent():
    """Handle user's privacy consent response"""
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        user_choice = data.get('user_choice', {})
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        # Handle consent with agentic system
        if agentic_system:
            consent_result = agentic_system.handle_privacy_consent(user_choice)
        else:
            consent_result = {
                "status": "success",
                "message": "Privacy settings updated"
            }
        
        # Update session based on consent choice
        agentic_mode = user_choice.get("choice") == "remember"
        updates = {
            'consent_given': True,
            'agentic_mode': agentic_mode,
            'consent_timestamp': datetime.now().isoformat()
        }
        
        if agentic_mode and consent_result.get("status") == "success":
            updates['user_id'] = consent_result.get('user_id')
        
        session_manager.update_session(session_id, updates)
        
        return jsonify({
            'success': True,
            'consent_result': consent_result,
            'agentic_mode': agentic_mode
        })
        
    except Exception as e:
        logger.error(f"Error handling privacy consent: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to handle privacy consent'
        }), 500

@app.route('/api/user/authenticate', methods=['POST'])
def authenticate_user():
    """Authenticate returning user"""
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        user_id = data.get('user_id')
        password = data.get('password')
        
        if not all([session_id, user_id, password]):
            return jsonify({
                'success': False,
                'error': 'Session ID, user ID, and password required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        # Authenticate with agentic system
        if agentic_system:
            auth_result = agentic_system.authenticate_returning_user(user_id, password)
        else:
            auth_result = {
                "status": "error",
                "message": "Agentic system not available"
            }
        
        if auth_result.get("status") == "success":
            session_manager.update_session(session_id, {
                'consent_given': True,
                'agentic_mode': True,
                'user_id': user_id,
                'authenticated': True
            })
        
        return jsonify({
            'success': auth_result.get("status") == "success",
            'auth_result': auth_result
        })
        
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return jsonify({
            'success': False,
            'error': 'Authentication failed'
        }), 500

@app.route('/api/session/consent', methods=['POST'])
def update_consent():
    """Legacy consent endpoint - redirects to privacy consent"""
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        consent = data.get('consent', False)
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        session_manager.update_session(session_id, {
            'consent_given': consent,
            'consent_timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'message': 'Consent updated',
            'privacy_consent_available': agentic_system is not None
        })
        
    except Exception as e:
        logger.error(f"Error updating consent: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to update consent'
        }), 500

@app.route('/api/chat/send', methods=['POST'])
def send_message():
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        include_video = data.get('include_video', False)  # Check if video is enabled
        
        if not session_id or not message:
            return jsonify({
                'success': False,
                'error': 'Session ID and message required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        if not session_data.get('consent_given', False):
            return jsonify({
                'success': False,
                'error': 'Consent required'
            }), 403
        
        # Analyze text
        analysis_results = None
        if text_analyzer:
            try:
                analysis_results = text_analyzer.analyze_text(message)
                logger.info(f"Text analysis completed for session {session_id}")
            except Exception as e:
                logger.error(f"Text analysis failed: {e}")
                analysis_results = _get_fallback_analysis(message)
        else:
            analysis_results = _get_fallback_analysis(message)
        
        # Analyze video if enabled and available
        video_analysis = None
        if include_video and video_agent:
            try:
                video_result = video_agent.get_latest_analysis()
                if video_result.get('success'):
                    video_analysis = video_result
                    logger.info(f"Video analysis included: {video_analysis.get('dominant_emotion')} detected")
                    
                    # Merge video emotion into text analysis
                    if analysis_results:
                        analysis_results['video_emotion'] = video_analysis.get('dominant_emotion')
                        analysis_results['video_confidence'] = video_analysis.get('confidence', 0.0)
                        analysis_results['faces_detected'] = video_analysis.get('faces_detected', 0)
            except Exception as e:
                logger.error(f"Video analysis failed: {e}")
                video_analysis = None
        
        # Generate response using agentic system if available and enabled
        therapy_response = None
        
        # Prepare context for video-enhanced response
        enhanced_context = {
            'text_analysis': analysis_results,
            'video_analysis': video_analysis,
            'has_video': video_analysis is not None and video_analysis.get('faces_detected', 0) > 0
        }
        
        if agentic_system and session_data.get('agentic_mode', False):
            try:
                therapy_response = agentic_system.generate_agentic_response(message, analysis_results)
                
                # Enhance response with video observation if available
                if enhanced_context['has_video']:
                    video_emotion = video_analysis.get('dominant_emotion', 'neutral')
                    video_confidence = video_analysis.get('confidence', 0.0)
                    
                    # Prepend video observation to response
                    emotion_observations = {
                        'happy': "I can see from your facial expression that you're feeling happy",
                        'sad': "I notice from your facial expression that you seem sad",
                        'angry': "I can see from your face that you might be feeling frustrated or angry",
                        'fear': "Your facial expression suggests you might be feeling anxious or worried",
                        'surprise': "You look surprised! ",
                        'neutral': "I'm looking at your facial expression",
                        'disgust': "Your expression suggests something is bothering you"
                    }
                    
                    observation = emotion_observations.get(video_emotion, f"I can see your {video_emotion} expression")
                    
                    if video_confidence > 0.6:
                        video_prefix = f"{observation}. "
                    else:
                        video_prefix = f"{observation}, though I'm not entirely certain. "
                    
                    therapy_response['content'] = video_prefix + therapy_response['content']
                
                logger.info(f"Agentic response generated for session {session_id}")
            except Exception as e:
                logger.error(f"Agentic response failed: {e}")
                # Fallback to regular therapy agent
                if therapy_agent:
                    therapy_response = therapy_agent.generate_response(
                        message, analysis_results, session_data['chat_history']
                    )
                    
                    # Add video observation to fallback response too
                    if enhanced_context['has_video']:
                        video_emotion = video_analysis.get('dominant_emotion', 'neutral')
                        video_confidence = video_analysis.get('confidence', 0.0)
                        
                        emotion_observations = {
                            'happy': "I can see you're smiling",
                            'sad': "I notice you look a bit down",
                            'angry': "I can see you might be feeling upset",
                            'fear': "You seem worried or anxious",
                            'surprise': "You look surprised",
                            'neutral': "I'm observing your expression",
                            'disgust': "Something seems to be troubling you"
                        }
                        
                        observation = emotion_observations.get(video_emotion, f"I notice your {video_emotion} expression")
                        therapy_response['content'] = f"{observation}. {therapy_response['content']}"
                else:
                    therapy_response = _get_fallback_response(message, analysis_results)
        elif therapy_agent:
            try:
                therapy_response = therapy_agent.generate_response(
                    message, analysis_results, session_data['chat_history']
                )
                
                # Add video observation to therapy response
                if enhanced_context['has_video']:
                    video_emotion = video_analysis.get('dominant_emotion', 'neutral')
                    video_confidence = video_analysis.get('confidence', 0.0)
                    
                    emotion_observations = {
                        'happy': "I can see from your facial expression that you're feeling positive",
                        'sad': "I notice from your face that you seem sad or down",
                        'angry': "I can see you might be feeling frustrated or angry right now",
                        'fear': "Your facial expression shows you might be anxious or worried",
                        'surprise': "You look surprised! ",
                        'neutral': "I'm watching your expression as we talk",
                        'disgust': "Your expression tells me something is bothering you"
                    }
                    
                    observation = emotion_observations.get(video_emotion, f"I notice your {video_emotion} expression")
                    
                    if video_confidence > 0.6:
                        video_prefix = f"{observation}. "
                    else:
                        video_prefix = f"{observation}, though I'm reading between the lines. "
                    
                    therapy_response['content'] = video_prefix + therapy_response['content']
                
                logger.info(f"Therapy response generated for session {session_id}")
            except Exception as e:
                logger.error(f"Therapy response failed: {e}")
                therapy_response = _get_fallback_response(message, analysis_results)
        else:
            therapy_response = _get_fallback_response(message, analysis_results)
            
            # Add video observation even to fallback
            if enhanced_context['has_video']:
                video_emotion = video_analysis.get('dominant_emotion', 'neutral')
                observation = f"I can see you look {video_emotion}. "
                therapy_response['content'] = observation + therapy_response['content']
        
        # Create message objects
        user_message = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis_results
        }
        
        assistant_message = {
            'role': 'assistant',
            'content': therapy_response['content'],
            'timestamp': datetime.now().isoformat(),
            'provider': therapy_response.get('provider', 'unknown'),
            'technique': therapy_response.get('technique', 'unknown'),
            'personalized': therapy_response.get('personalized', False),
            'video_enhanced': video_analysis is not None  # Flag to show video was used
        }
        
        # Add agentic features if available
        agentic_features = {}
        if therapy_response.get('proactive_suggestion'):
            agentic_features['proactive_suggestion'] = therapy_response['proactive_suggestion']
        if therapy_response.get('goal_progress'):
            agentic_features['goal_progress'] = therapy_response['goal_progress']
        
        # Add video analysis to agentic features if available
        if video_analysis and video_analysis.get('faces_detected', 0) > 0:
            agentic_features['video_emotion_detected'] = {
                'emotion': video_analysis.get('dominant_emotion'),
                'confidence': video_analysis.get('confidence'),
                'therapeutic_suggestion': video_analysis.get('therapeutic_suggestion')
            }
        
        # Update session
        session_data['chat_history'].extend([user_message, assistant_message])
        session_data['stats']['message_count'] += 1
        session_duration = (datetime.now() - session_data['created']).total_seconds() / 60
        session_data['stats']['session_duration'] = int(session_duration)
        
        session_manager.update_session(session_id, session_data)
        
        # Check for crisis
        crisis_detected = analysis_results.get('crisis_classification', 'LOW') in ['HIGH', 'CRITICAL']
        if crisis_detected:
            _log_crisis_event(session_id, analysis_results)
        
        response_data = {
            'success': True,
            'assistant_response': assistant_message,
            'analysis': analysis_results,
            'video_analysis': video_analysis,  # Include video data in response
            'crisis_detected': crisis_detected,
            'session_stats': session_data['stats'],
            'agentic_mode': session_data.get('agentic_mode', False)
        }
        
        # Add agentic features to response
        if agentic_features:
            response_data['agentic_features'] = agentic_features
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process message'
        }), 500

@app.route('/api/goals/create', methods=['POST'])
def create_goal():
    """Create a new therapeutic goal"""
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        target_days = data.get('target_days', 30)
        
        if not session_id or not title:
            return jsonify({
                'success': False,
                'error': 'Session ID and goal title required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        if not session_data.get('agentic_mode', False):
            return jsonify({
                'success': False,
                'error': 'Goal tracking requires persistent memory mode'
            }), 403
        
        # Create goal using agentic system
        if agentic_system:
            goal_result = agentic_system.create_user_goal(title, description, target_days)
        else:
            goal_result = {"error": "Agentic system not available"}
        
        return jsonify({
            'success': 'error' not in goal_result,
            'goal_result': goal_result
        })
        
    except Exception as e:
        logger.error(f"Error creating goal: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to create goal'
        }), 500

@app.route('/api/goals/list', methods=['GET'])
def list_goals():
    """Get user's active goals"""
    try:
        ensure_initialized()
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        if not session_data.get('agentic_mode', False):
            return jsonify({
                'success': False,
                'error': 'Goal tracking requires persistent memory mode'
            }), 403
        
        # Get goals from agentic system
        if agentic_system and session_data.get('user_id'):
            goals = agentic_system.memory_manager.get_active_goals(session_data['user_id'])
            goals_data = [goal.__dict__ for goal in goals]
        else:
            goals_data = []
        
        return jsonify({
            'success': True,
            'goals': goals_data
        })
        
    except Exception as e:
        logger.error(f"Error listing goals: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to list goals'
        }), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Get user dashboard with goals, progress, and insights"""
    try:
        ensure_initialized()
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        if not session_data.get('agentic_mode', False):
            return jsonify({
                'success': False,
                'message': 'Dashboard requires persistent memory mode'
            })
        
        # Get dashboard from agentic system
        if agentic_system:
            dashboard = agentic_system.get_user_dashboard()
        else:
            dashboard = {"message": "Agentic system not available"}
        
        return jsonify({
            'success': True,
            'dashboard': dashboard
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get dashboard'
        }), 500

@app.route('/api/user/delete', methods=['POST'])
def delete_user_data():
    """Delete all user data"""
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        # Delete data using agentic system
        if agentic_system:
            deletion_result = agentic_system.delete_all_user_data()
        else:
            deletion_result = {"message": "No data to delete"}
        
        # Reset session to privacy mode
        session_manager.update_session(session_id, {
            'agentic_mode': False,
            'user_id': None,
            'authenticated': False
        })
        
        return jsonify({
            'success': True,
            'deletion_result': deletion_result
        })
        
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to delete user data'
        }), 500

@app.route('/api/session/settings', methods=['POST'])
def update_settings():
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        settings = data.get('settings', {})
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        session_data['settings'].update(settings)
        session_manager.update_session(session_id, session_data)
        
        return jsonify({
            'success': True,
            'settings': session_data['settings']
        })
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to update settings'
        }), 500

@app.route('/api/session/stats/<session_id>')
def get_session_stats(session_id):
    try:
        ensure_initialized()
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        chat_history = session_data['chat_history']
        user_messages = [msg for msg in chat_history if msg['role'] == 'user']
        
        recent_emotions = []
        for msg in user_messages[-3:]:
            analysis = msg.get('analysis', {})
            emotion = analysis.get('dominant_emotion')
            if emotion and emotion != 'neutral':
                recent_emotions.append(emotion)
        
        dominant_emotion = recent_emotions[-1] if recent_emotions else 'neutral'
        
        stats = session_data['stats'].copy()
        stats.update({
            'dominant_emotion': dominant_emotion,
            'duration_minutes': stats['session_duration'],
            'agentic_mode': session_data.get('agentic_mode', False),
            'user_authenticated': session_data.get('authenticated', False)
        })
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get session stats'
        }), 500

@app.route('/api/session/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    try:
        ensure_initialized()
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404
        
        session_data['chat_history'] = []
        session_data['stats'] = {
            'message_count': 0,
            'session_duration': 0
        }
        session_data['consent_given'] = False
        session_data['agentic_mode'] = False
        session_data['user_id'] = None
        session_data['authenticated'] = False
        
        session_manager.update_session(session_id, session_data)
        
        return jsonify({
            'success': True,
            'message': 'Session reset successfully'
        })
        
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to reset session'
        }), 500

@app.route('/api/providers/status')
def get_provider_status():
    try:
        ensure_initialized()
        status = {
            'providers': {'ollama': False, 'groq': False},
            'primary': 'rule_based',
            'last_checked': datetime.now().isoformat(),
            'agentic_available': agentic_system is not None
        }
        
        if therapy_agent:
            agent_status = therapy_agent.get_provider_status()
            status.update(agent_status)
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting provider status: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get provider status'
        }), 500

@app.route('/api/providers/test')
def test_providers():
    try:
        ensure_initialized()
        results = {
            'ollama': {'available': False, 'reason': 'agent_not_initialized'},
            'groq': {'available': False, 'reason': 'agent_not_initialized'},
            'fallback': {'available': True, 'response_length': 100},
            'agentic_system': {'available': agentic_system is not None}
        }
        
        if therapy_agent:
            agent_results = therapy_agent.test_providers()
            results.update(agent_results)
        
        return jsonify({
            'success': True,
            'test_results': results
        })
        
    except Exception as e:
        logger.error(f"Error testing providers: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to test providers'
        }), 500

# ===== VOICE AGENT ENDPOINTS =====

@app.route('/api/voice/status')
def get_voice_status():
    """Get voice agent status"""
    try:
        ensure_initialized()
        if voice_agent:
            status = voice_agent.get_voice_status()
            capabilities = voice_agent.get_voice_capabilities()
            return jsonify({
                'success': True,
                'status': status,
                'capabilities': capabilities
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Voice agent not available'
            }), 503
    except Exception as e:
        logger.error(f"Error getting voice status: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get voice status'
        }), 500

@app.route('/api/voice/speak', methods=['POST'])
def speak_text():
    """Convert text to speech"""
    try:
        ensure_initialized()
        if not voice_agent:
            return jsonify({
                'success': False,
                'error': 'Voice agent not available'
            }), 503

        data = request.get_json()
        text = data.get('text', '').strip()
        async_mode = data.get('async', True)

        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400

        result = voice_agent.speak_text(text, async_mode=async_mode)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return jsonify({
            'success': False,
            'error': 'Text-to-speech failed'
        }), 500

@app.route('/api/voice/listen', methods=['POST'])
def listen_for_speech():
    """Listen for speech input and convert to text"""
    try:
        ensure_initialized()
        if not voice_agent:
            return jsonify({
                'success': False,
                'error': 'Voice agent not available'
            }), 503

        data = request.get_json() or {}
        duration = data.get('duration')  # Optional duration limit

        if duration:
            result = voice_agent.capture_audio(duration=duration)
        else:
            result = voice_agent.process_voice_interaction()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in speech recognition: {e}")
        return jsonify({
            'success': False,
            'error': 'Speech recognition failed'
        }), 500

# ===== VIDEO AGENT ENDPOINTS =====

@app.route('/api/video/status')
def get_video_status():
    """Get video agent status"""
    try:
        ensure_initialized()
        if video_agent:
            status = video_agent.get_video_status()
            capabilities = video_agent.get_video_capabilities()
            return jsonify({
                'success': True,
                'status': status,
                'capabilities': capabilities
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Video agent not available'
            }), 503
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get video status'
        }), 500

@app.route('/api/video/start', methods=['POST'])
def start_video_camera():
    """Start video camera"""
    try:
        ensure_initialized()
        if not video_agent:
            return jsonify({
                'success': False,
                'error': 'Video agent not available'
            }), 503

        result = video_agent.start_camera()
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to start camera'
        }), 500

@app.route('/api/video/analyze', methods=['POST'])
def analyze_video_emotion():
    """Analyze current video frame for emotions"""
    try:
        ensure_initialized()
        if not video_agent:
            return jsonify({
                'success': False,
                'error': 'Video agent not available'
            }), 503

        result = video_agent.analyze_current_frame()
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        return jsonify({
            'success': False,
            'error': 'Video analysis failed'
        }), 500

@app.route('/api/video/stream')
def video_stream():
    """Video stream endpoint for live camera feed"""
    try:
        ensure_initialized()
        if not video_agent:
            return jsonify({
                'success': False,
                'error': 'Video agent not available'
            }), 503

        def generate_video():
            """Generate video frames"""
            while True:
                try:
                    frame_result = video_agent.capture_and_encode_frame()
                    if frame_result['success']:
                        yield f"data: {frame_result['frame_base64']}\n\n"
                    else:
                        yield f"data: error\n\n"
                        break
                except Exception as e:
                    logger.error(f"Error generating video frame: {e}")
                    break
                import time
                time.sleep(0.1)  # 10 FPS

        return Response(generate_video(), mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error starting video stream: {e}")
        return jsonify({
            'success': False,
            'error': 'Video stream failed'
        }), 500

@app.route('/api/video/stop', methods=['POST'])
def stop_video_camera():
    """Stop video camera"""
    try:
        ensure_initialized()
        if not video_agent:
            return jsonify({
                'success': False,
                'error': 'Video agent not available'
            }), 503

        result = video_agent.stop_camera()
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to stop camera'
        }), 500

# ===== MULTIMODAL CHAT ENDPOINT =====

@app.route('/api/chat/multimodal', methods=['POST'])
def send_multimodal_message():
    """Enhanced chat endpoint with voice and video integration"""
    try:
        ensure_initialized()
        data = request.get_json()
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        include_voice = data.get('include_voice', False)
        include_video = data.get('include_video', False)

        if not session_id or not message:
            return jsonify({
                'success': False,
                'error': 'Session ID and message required'
            }), 400

        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 404

        if not session_data.get('consent_given', False):
            return jsonify({
                'success': False,
                'error': 'Consent required'
            }), 403

        # Get video emotion analysis if requested
        video_analysis = None
        if include_video and video_agent:
            video_result = video_agent.analyze_current_frame()
            if video_result["success"]:
                video_analysis = {
                    'dominant_emotion': video_result.get('dominant_emotion'),
                    'confidence': video_result.get('confidence'),
                    'faces_detected': video_result.get('faces_detected'),
                    'therapy_priority': video_result.get('therapy_priority'),
                    'therapeutic_suggestion': video_result.get('therapeutic_suggestion')
                }

        # Analyze text
        analysis_results = None
        if text_analyzer:
            try:
                analysis_results = text_analyzer.analyze_text(message)

                # Enhance analysis with video data
                if video_analysis:
                    analysis_results['video_emotion'] = video_analysis
                    # Combine text and video emotions for better analysis
                    if video_analysis['confidence'] > 0.6:
                        analysis_results['multimodal_emotion'] = video_analysis['dominant_emotion']
                        analysis_results['multimodal_confidence'] = video_analysis['confidence']

                logger.info(f"Multimodal analysis completed for session {session_id}")
            except Exception as e:
                logger.error(f"Text analysis failed: {e}")
                analysis_results = _get_fallback_analysis(message)
                if video_analysis:
                    analysis_results['video_emotion'] = video_analysis
        else:
            analysis_results = _get_fallback_analysis(message)
            if video_analysis:
                analysis_results['video_emotion'] = video_analysis

        # Generate response
        therapy_response = None
        if agentic_system and session_data.get('agentic_mode', False):
            try:
                therapy_response = agentic_system.generate_agentic_response(message, analysis_results)
            except Exception as e:
                logger.error(f"Agentic response failed: {e}")
                if therapy_agent:
                    therapy_response = therapy_agent.generate_response(
                        message, analysis_results, session_data['chat_history']
                    )
                else:
                    therapy_response = _get_fallback_response(message, analysis_results)
        elif therapy_agent:
            try:
                therapy_response = therapy_agent.generate_response(
                    message, analysis_results, session_data['chat_history']
                )
            except Exception as e:
                logger.error(f"Therapy response failed: {e}")
                therapy_response = _get_fallback_response(message, analysis_results)
        else:
            therapy_response = _get_fallback_response(message, analysis_results)

        # Convert response to speech if requested
        voice_response = None
        if include_voice and voice_agent and therapy_response:
            try:
                voice_result = voice_agent.speak_text(therapy_response['content'], async_mode=True)
                if voice_result['success']:
                    voice_response = {
                        'speech_generated': True,
                        'speech_length': voice_result.get('text_length', 0),
                        'async_mode': voice_result.get('async', True)
                    }
            except Exception as e:
                logger.error(f"Voice synthesis failed: {e}")
                voice_response = {'speech_generated': False, 'error': str(e)}

        # Create message objects
        user_message = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis_results,
            'multimodal': {
                'voice_input': False,  # This would be True if message came from voice
                'video_analysis': video_analysis is not None
            }
        }

        assistant_message = {
            'role': 'assistant',
            'content': therapy_response['content'],
            'timestamp': datetime.now().isoformat(),
            'provider': therapy_response.get('provider', 'unknown'),
            'technique': therapy_response.get('technique', 'unknown'),
            'personalized': therapy_response.get('personalized', False),
            'multimodal': {
                'voice_output': voice_response is not None,
                'video_aware': video_analysis is not None
            }
        }

        # Update session
        session_data['chat_history'].extend([user_message, assistant_message])
        session_data['stats']['message_count'] += 1
        session_duration = (datetime.now() - session_data['created']).total_seconds() / 60
        session_data['stats']['session_duration'] = int(session_duration)

        session_manager.update_session(session_id, session_data)

        # Check for crisis
        crisis_detected = analysis_results.get('crisis_classification', 'LOW') in ['HIGH', 'CRITICAL']
        if crisis_detected:
            _log_crisis_event(session_id, analysis_results)

        response_data = {
            'success': True,
            'assistant_response': assistant_message,
            'analysis': analysis_results,
            'crisis_detected': crisis_detected,
            'session_stats': session_data['stats'],
            'agentic_mode': session_data.get('agentic_mode', False),
            'multimodal_features': {
                'voice_synthesis': voice_response,
                'video_emotion': video_analysis
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing multimodal message: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process multimodal message'
        }), 500

def _get_fallback_analysis(message: str) -> Dict[str, Any]:
    """Fallback analysis when text analyzer is unavailable"""
    return {
        'input_text': message,
        'timestamp': datetime.now().isoformat(),
        'word_count': len(message.split()),
        'dominant_emotion': 'neutral',
        'emotion_confidence': 0.3,
        'sentiment': 'neutral',
        'sentiment_score': 0.5,
        'crisis_level': 0.0,
        'crisis_classification': 'LOW',
        'crisis_indicators': [],
        'mental_health_topics': [],
        'primary_topic': 'general',
        'overall_risk_score': 0.0,
        'risk_level': 'LOW',
        'suggested_techniques': ['Active Listening'],
        'analysis_confidence': 0.3,
        'analysis_method': 'fallback'
    }

def _get_fallback_response(message: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback response when therapy agents are unavailable"""
    fallback_responses = [
        "I hear what you're sharing, and I want you to know that I'm here to listen. How are you feeling right now?",
        "Thank you for opening up about this. It takes courage to share difficult feelings. What kind of support would be most helpful?",
        "I can sense this is important to you. Sometimes just talking through our thoughts and feelings can be helpful.",
        "What you're experiencing sounds challenging. Have you been able to talk to anyone else about this?"
    ]
    return {
        'content': random.choice(fallback_responses),
        'provider': 'fallback',
        'technique': 'active_listening',
        'confidence': 0.5,
        'timestamp': datetime.now().isoformat()
    }
    



def _log_crisis_event(session_id: str, analysis: Dict[str, Any]):
    """Log crisis events for monitoring"""
    try:
        os.makedirs('logs', exist_ok=True)
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id[:8] + '***',
            'crisis_level': analysis.get('crisis_level', 0.0),
            'crisis_classification': analysis.get('crisis_classification', 'UNKNOWN'),
            'indicators': analysis.get('crisis_indicators', []),
            'action_taken': 'resources_provided'
        }
        with open('logs/crisis_events.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        logger.warning(f"Crisis event logged for session {session_id[:8]}***")
    except Exception as e:
        logger.error(f"Failed to log crisis event: {e}")

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.before_request
def pre_request_housekeeping():
    ensure_initialized()
    if random.random() < 0.01:
        session_manager.cleanup_expired_sessions()

if __name__ == '__main__':
    if os.environ.get('FLASK_ENV') == 'development':
        logging.getLogger().setLevel(logging.DEBUG)
    ensure_initialized()
    app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )