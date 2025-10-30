"""
Flask API Routes for Crisis Counselling Mode
Provides REST endpoints for crisis detection and compassionate responses
"""

from flask import Blueprint, request, jsonify
import logging
from typing import Dict, Any
from datetime import datetime

# Import crisis components
try:
    from crisis_counselling_mode import CrisisCounsellingMode
    from text_analyzer import TextAnalyzer
    CRISIS_MODE_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import crisis components: {e}")
    CRISIS_MODE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create Blueprint
crisis_bp = Blueprint('crisis', __name__, url_prefix='/api/crisis')

# Initialize components (will be set by app factory)
crisis_counselor = None
text_analyzer = None


def init_crisis_api(crisis_counselor_instance: CrisisCounsellingMode = None,
                   text_analyzer_instance: TextAnalyzer = None):
    """Initialize the crisis API with component instances"""
    global crisis_counselor, text_analyzer

    if crisis_counselor_instance:
        crisis_counselor = crisis_counselor_instance
        logger.info("✓ Crisis API initialized with Crisis Counselor")

    if text_analyzer_instance:
        text_analyzer = text_analyzer_instance
        logger.info("✓ Crisis API initialized with Text Analyzer")


@crisis_bp.route('/status', methods=['GET'])
def get_crisis_status():
    """Get crisis counselling system status"""
    return jsonify({
        'success': True,
        'available': crisis_counselor is not None,
        'text_analyzer_available': text_analyzer is not None,
        'timestamp': datetime.now().isoformat(),
        'features': {
            'crisis_detection': True,
            'emotional_adaptation': True,
            'coping_strategies': True,
            'professional_resources': True,
            'llm_integration': True
        }
    })


@crisis_bp.route('/analyze', methods=['POST'])
def analyze_crisis():
    """
    Analyze a message for crisis indicators

    Request body:
    {
        "message": "User's message text"
    }

    Returns:
    {
        "success": true,
        "crisis_detected": true,
        "primary_crisis": "grief_loss",
        "severity": "high",
        "emotional_tone": "acute_distress",
        "immediate_response_needed": false,
        "all_detected_crises": [...]
    }
    """
    try:
        if not crisis_counselor:
            return jsonify({
                'success': False,
                'error': 'Crisis counseling mode not available'
            }), 503

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: message'
            }), 400

        user_message = data['message']

        # Get text analysis if analyzer is available
        text_analysis = None
        if text_analyzer:
            text_analysis = text_analyzer.analyze_text(user_message)

        # Analyze crisis context
        crisis_analysis = crisis_counselor.analyze_crisis_context(
            user_message=user_message,
            text_analysis=text_analysis
        )

        return jsonify({
            'success': True,
            'crisis_detected': crisis_analysis['severity'] != 'low',
            'primary_crisis': crisis_analysis['primary_crisis'].value,
            'severity': crisis_analysis['severity'],
            'emotional_tone': crisis_analysis['emotional_tone'].value,
            'immediate_response_needed': crisis_analysis['immediate_response_needed'],
            'all_detected_crises': [
                {
                    'type': c['type'].value,
                    'confidence': c['confidence'],
                    'severity': c['severity']
                }
                for c in crisis_analysis['all_detected_crises']
            ],
            'context': {
                'mentioned_topics': list(crisis_analysis['context']['mentioned_topics']),
                'support_system_mentioned': crisis_analysis['context']['support_system'] is not None
            }
        })

    except Exception as e:
        logger.error(f"Error in crisis analysis: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500


@crisis_bp.route('/respond', methods=['POST'])
def generate_crisis_response():
    """
    Generate a compassionate crisis response

    Request body:
    {
        "message": "User's message text",
        "chat_history": [...],  // Optional
        "llm_response": "..."  // Optional pre-generated LLM response
    }

    Returns:
    {
        "success": true,
        "response": "Compassionate response text...",
        "crisis_type": "grief_loss",
        "severity": "high",
        "immediate_response_needed": false,
        "coping_strategies": {...},
        "resources": {...}
    }
    """
    try:
        if not crisis_counselor:
            return jsonify({
                'success': False,
                'error': 'Crisis counseling mode not available'
            }), 503

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: message'
            }), 400

        user_message = data['message']
        llm_response = data.get('llm_response')

        # Get text analysis
        text_analysis = None
        if text_analyzer:
            text_analysis = text_analyzer.analyze_text(user_message)

        # Analyze crisis
        crisis_analysis = crisis_counselor.analyze_crisis_context(
            user_message=user_message,
            text_analysis=text_analysis
        )

        # Generate response
        response = crisis_counselor.generate_crisis_response(
            user_message=user_message,
            crisis_analysis=crisis_analysis,
            llm_response=llm_response
        )

        return jsonify({
            'success': True,
            'response': response['response'],
            'crisis_type': response['crisis_type'],
            'severity': response['severity'],
            'emotional_tone': response['emotional_tone'],
            'immediate_response_needed': response['immediate_response_needed'],
            'coping_strategies': response['coping_strategies'],
            'resources': response['resources'],
            'conversation_context': response['conversation_context']
        })

    except Exception as e:
        logger.error(f"Error generating crisis response: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500


@crisis_bp.route('/chat', methods=['POST'])
def crisis_chat():
    """
    Complete crisis counseling chat endpoint (analyze + respond)

    Request body:
    {
        "message": "User's message text",
        "session_id": "optional-session-id"
    }

    Returns:
    {
        "success": true,
        "response": "Full compassionate response...",
        "analysis": {...},
        "coping_strategies": {...},
        "resources": {...}
    }
    """
    try:
        if not crisis_counselor:
            return jsonify({
                'success': False,
                'error': 'Crisis counseling mode not available'
            }), 503

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: message'
            }), 400

        user_message = data['message']

        # Full analysis and response
        text_analysis = None
        if text_analyzer:
            text_analysis = text_analyzer.analyze_text(user_message)

        crisis_analysis = crisis_counselor.analyze_crisis_context(
            user_message=user_message,
            text_analysis=text_analysis
        )

        response = crisis_counselor.generate_crisis_response(
            user_message=user_message,
            crisis_analysis=crisis_analysis
        )

        return jsonify({
            'success': True,
            'response': response['response'],
            'analysis': {
                'crisis_type': response['crisis_type'],
                'severity': response['severity'],
                'emotional_tone': response['emotional_tone'],
                'immediate_response_needed': response['immediate_response_needed'],
                'emotion': text_analysis.get('dominant_emotion') if text_analysis else None,
                'sentiment': text_analysis.get('sentiment') if text_analysis else None
            },
            'coping_strategies': response['coping_strategies'],
            'resources': response['resources'],
            'conversation_context': response['conversation_context'],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in crisis chat: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500


@crisis_bp.route('/conversation/summary', methods=['GET'])
def get_conversation_summary():
    """
    Get summary of current crisis counseling conversation

    Returns:
    {
        "success": true,
        "summary": {...}
    }
    """
    try:
        if not crisis_counselor:
            return jsonify({
                'success': False,
                'error': 'Crisis counseling mode not available'
            }), 503

        summary = crisis_counselor.get_conversation_summary()

        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500


@crisis_bp.route('/resources', methods=['GET'])
def get_crisis_resources():
    """
    Get crisis resources and hotlines

    Query params:
    - crisis_type: Filter resources by crisis type (optional)

    Returns:
    {
        "success": true,
        "resources": {...}
    }
    """
    try:
        if not crisis_counselor:
            return jsonify({
                'success': False,
                'error': 'Crisis counseling mode not available'
            }), 503

        # Get all resources
        all_resources = crisis_counselor.professional_resources

        crisis_type = request.args.get('crisis_type')

        if crisis_type:
            # Filter for specific crisis type
            resources = crisis_counselor._get_relevant_resources(
                getattr(crisis_counselor.crisis_patterns, crisis_type.upper(), None)
            )
        else:
            resources = all_resources

        return jsonify({
            'success': True,
            'resources': resources
        })

    except Exception as e:
        logger.error(f"Error getting resources: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500


@crisis_bp.route('/coping-strategies', methods=['GET'])
def get_coping_strategies():
    """
    Get coping strategies for a specific crisis type

    Query params:
    - crisis_type: Type of crisis (required)
    - level: immediate, short_term, or long_term (optional)

    Returns:
    {
        "success": true,
        "strategies": {...}
    }
    """
    try:
        if not crisis_counselor:
            return jsonify({
                'success': False,
                'error': 'Crisis counseling mode not available'
            }), 503

        crisis_type = request.args.get('crisis_type')
        level = request.args.get('level', 'all')

        if not crisis_type:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: crisis_type'
            }), 400

        # Get strategies
        from crisis_counselling_mode import CrisisType

        try:
            crisis_enum = CrisisType[crisis_type.upper()]
        except KeyError:
            return jsonify({
                'success': False,
                'error': f'Invalid crisis type: {crisis_type}'
            }), 400

        all_strategies = crisis_counselor.coping_strategies.get(crisis_enum, {})

        if level != 'all' and level in all_strategies:
            strategies = {level: all_strategies[level]}
        else:
            strategies = all_strategies

        return jsonify({
            'success': True,
            'crisis_type': crisis_type,
            'strategies': strategies
        })

    except Exception as e:
        logger.error(f"Error getting coping strategies: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500


@crisis_bp.route('/crisis-types', methods=['GET'])
def get_crisis_types():
    """
    Get list of all supported crisis types

    Returns:
    {
        "success": true,
        "crisis_types": [...]
    }
    """
    try:
        from crisis_counselling_mode import CrisisType

        crisis_types = [
            {
                'value': ct.value,
                'name': ct.name,
                'description': _get_crisis_description(ct)
            }
            for ct in CrisisType
        ]

        return jsonify({
            'success': True,
            'crisis_types': crisis_types
        })

    except Exception as e:
        logger.error(f"Error getting crisis types: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500


def _get_crisis_description(crisis_type) -> str:
    """Get human-readable description for crisis type"""
    descriptions = {
        'GRIEF_LOSS': 'Death of loved ones, bereavement, mourning',
        'RELATIONSHIP_BREAKUP': 'End of romantic relationships, heartbreak',
        'COVID_STRESS': 'Pandemic-related anxiety, isolation, job loss',
        'DEPRESSION': 'Persistent sadness, hopelessness, lack of energy',
        'ANXIETY_PANIC': 'Panic attacks, overwhelming fear, constant worry',
        'TRAUMA': 'PTSD, assault, traumatic experiences',
        'SUICIDAL_IDEATION': 'Thoughts of suicide or self-harm',
        'SELF_HARM': 'Urges to physically hurt oneself',
        'ISOLATION_LONELINESS': 'Social disconnection, feeling alone',
        'HEALTH_CRISIS': 'Serious illness diagnosis, chronic health issues',
        'FINANCIAL_STRESS': 'Debt, job loss, money problems',
        'FAMILY_CONFLICT': 'Family-related distress and conflicts',
        'GENERAL_DISTRESS': 'Undefined or mixed emotional struggles'
    }
    return descriptions.get(crisis_type.name, 'General crisis situation')


# Error handlers
@crisis_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@crisis_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
