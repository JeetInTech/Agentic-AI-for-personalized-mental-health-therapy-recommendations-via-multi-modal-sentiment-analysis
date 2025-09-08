"""
Simplified Flask Backend for AI Therapy System
Phase 1: Text-only analysis with robust error handling
"""

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import uuid
import logging
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Any, Optional
import threading
import random
from dotenv import load_dotenv

# Import our simplified modules
from text_analyzer import TextAnalyzer
from therapy_agent import TherapyAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
CORS(app)

text_analyzer = None
therapy_agent = None

active_sessions = {}

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
    global text_analyzer, therapy_agent
    
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
    
    if text_analyzer is None or therapy_agent is None:
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
            'therapy_agent': therapy_agent is not None
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

@app.route('/api/session/consent', methods=['POST'])
def update_consent():
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
            'message': 'Consent updated'
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
        
        therapy_response = None
        if therapy_agent and analysis_results:
            try:
                therapy_response = therapy_agent.generate_response(
                    message, 
                    analysis_results, 
                    session_data['chat_history']
                )
                logger.info(f"Therapy response generated for session {session_id}")
            except Exception as e:
                logger.error(f"Therapy response failed: {e}")
                therapy_response = _get_fallback_response(message, analysis_results)
        else:
            therapy_response = _get_fallback_response(message, analysis_results)
        
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
            'technique': therapy_response.get('technique', 'unknown')
        }
        
        session_data['chat_history'].extend([user_message, assistant_message])
        
        session_data['stats']['message_count'] += 1
        session_duration = (datetime.now() - session_data['created']).total_seconds() / 60
        session_data['stats']['session_duration'] = int(session_duration)
        
        session_manager.update_session(session_id, session_data)
        
        crisis_detected = analysis_results.get('crisis_classification', 'LOW') in ['HIGH', 'CRITICAL']
        
        if crisis_detected:
            _log_crisis_event(session_id, analysis_results)
        
        return jsonify({
            'success': True,
            'assistant_response': assistant_message,
            'analysis': analysis_results,
            'crisis_detected': crisis_detected,
            'session_stats': session_data['stats']
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process message'
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
            'duration_minutes': stats['session_duration']
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
        if therapy_agent:
            status = therapy_agent.get_provider_status()
        else:
            status = {
                'providers': {'ollama': False, 'groq': False},
                'primary': 'rule_based',
                'last_checked': datetime.now().isoformat()
            }
        
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
        if therapy_agent:
            results = therapy_agent.test_providers()
        else:
            results = {
                'ollama': {'available': False, 'reason': 'agent_not_initialized'},
                'groq': {'available': False, 'reason': 'agent_not_initialized'},
                'fallback': {'available': True, 'response_length': 100}
            }
        
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

def _get_fallback_analysis(message: str) -> Dict[str, Any]:
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
