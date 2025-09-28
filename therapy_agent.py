"""
Phase 1: Ollama → Groq → Rule-based fallback system
"""

import logging
import json
import os
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapyAgent:
    """
    Simplified therapy agent with reliable LLM integration and comprehensive fallbacks
    """
    
    def __init__(self, config_path="config.json"):
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.ollama_url = config.get("llm", {}).get("ollama_url", "http://localhost:11434")
            self.groq_api_key = config.get("llm", {}).get("groq_api_key") or os.getenv("GROQ_API_KEY")
            self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
            
            # Groq configuration from config
            self.groq_model = "llama-3.3-70b-versatile"  # Current supported model
            self.temperature = config.get("llm", {}).get("temperature", 0.7)
            self.max_tokens = config.get("llm", {}).get("max_tokens", 300)
            
            logger.info(f"Configuration loaded from {config_path}")
            if self.groq_api_key:
                logger.info("Groq API key loaded from config")
            
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults and environment variables")
            self.ollama_url = "http://localhost:11434"
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
            self.groq_model = "mixtral-8x7b-32768"
            self.temperature = 0.7
            self.max_tokens = 300
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Use defaults
            self.ollama_url = "http://localhost:11434"
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
            self.groq_model = "mixtral-8x7b-32768"
            self.temperature = 0.7
            self.max_tokens = 300
        
        # Provider status
        self.provider_status = {
            'ollama': False,
            'groq': False
        }
        
        # Therapeutic techniques database
        self.techniques = {
            'cognitive_restructuring': {
                'description': 'Identifying and challenging negative thought patterns',
                'prompts': [
                    "Let's examine that thought. What evidence supports it, and what evidence challenges it?",
                    "Can you think of a more balanced way to view this situation?",
                    "What would you tell a friend who had this same thought?"
                ]
            },
            'behavioral_activation': {
                'description': 'Increasing engagement in meaningful activities',
                'prompts': [
                    "What activities used to bring you joy or satisfaction?",
                    "What's one small step you could take today toward something meaningful?",
                    "How might you break this goal into smaller, manageable steps?"
                ]
            },
            'mindfulness': {
                'description': 'Present-moment awareness and acceptance',
                'prompts': [
                    "Let's focus on what you're experiencing right now. What do you notice?",
                    "Can you observe these feelings without judging them as good or bad?",
                    "What sensations do you notice in your body right now?"
                ]
            },
            'grounding': {
                'description': 'Techniques to manage overwhelming emotions',
                'prompts': [
                    "Let's try the 5-4-3-2-1 technique. Name 5 things you can see around you.",
                    "Take a slow, deep breath with me. Focus on the feeling of air entering and leaving your lungs.",
                    "Notice your feet on the ground. Feel that connection to the earth beneath you."
                ]
            }
        }
        
        # Crisis responses
        self.crisis_responses = {
            'HIGH': [
                "I'm very concerned about what you've shared. Your safety is important. Please consider reaching out to a crisis hotline at 988 or emergency services at 911.",
                "Thank you for trusting me with these difficult feelings. Right now, I want to make sure you're safe. The 988 Suicide & Crisis Lifeline is available 24/7.",
                "These feelings sound overwhelming. Please know that help is available. You can call 988 for immediate support or 911 if you're in immediate danger."
            ],
            'MODERATE': [
                "I hear that you're going through a really difficult time. Have you considered speaking with a mental health professional?",
                "These feelings are significant and deserve attention. It might be helpful to connect with a therapist or counselor.",
                "What support systems do you have available? Sometimes reaching out to a trusted friend or family member can help."
            ]
        }
        
        # Enhanced fallback responses with actionable suggestions
        self.actionable_responses = {
            'breakup': [
                "Breakups are incredibly painful. Here are some things that might help: 1) Allow yourself to grieve - it's a real loss, 2) Reach out to supportive friends/family, 3) Focus on self-care activities you enjoy, 4) Consider limiting social media if it's triggering. What feels most manageable to try first?",
                "Going through a breakup is one of life's most difficult experiences. Some concrete steps: Write in a journal about your feelings, try a 10-minute walk daily, call a friend who makes you laugh. Which of these resonates with you?",
                "Breakup recovery takes time, but there are ways to help yourself heal: Create new routines, practice the 'no contact' rule if possible, engage in activities that make you feel accomplished. What's one small step you could take today?"
            ],
            'progress_sharing': [
                "It sounds like you're taking some positive steps, which takes real courage. How did that feel for you? What worked well, and what felt challenging?",
                "I'm glad to hear you're trying different approaches. That shows real strength. What did you notice about yourself during that experience?",
                "Thank you for sharing your efforts with me. It's important to acknowledge when we take steps toward healing. What would feel like a natural next step from here?"
            ],
            'seeking_next_steps': [
                "Since you're ready to explore more options, let's think about what might build on what you've already tried. What area of your life feels like it needs the most attention right now?",
                "It sounds like you're motivated to continue working on this, which is wonderful. Consider: What time of day do you feel most capable? What activities have given you even small moments of relief?",
                "Moving forward, it might help to focus on: Building consistent daily structure, strengthening your support network, or developing healthy coping strategies. Which area feels most important to you?"
            ],
            'depression': [
                "Depression can make everything feel overwhelming. Small, concrete steps can help: 1) Try to maintain a sleep schedule, 2) Get 10 minutes of sunlight daily, 3) Do one small task that gives you a sense of accomplishment. What feels possible for you today?",
                "When depression hits, structure can be helpful. Consider: Setting one small daily goal, reaching out to one person, doing 5 minutes of movement. You don't have to do everything - pick what feels manageable.",
                "Depression lies to us about our worth and future. Combat this with: Daily self-compassion practice, connecting with others even briefly, engaging in one meaningful activity. Which approach feels right for you?"
            ],
            'anxiety': [
                "Anxiety can feel overwhelming, but there are concrete tools that help: 1) Practice the 4-7-8 breathing technique, 2) Use the 5-4-3-2-1 grounding method, 3) Write down your worries for 10 minutes. Which would you like to try?",
                "For anxiety management, try: Progressive muscle relaxation, limiting caffeine, breaking big worries into smaller actionable steps. What feels most doable right now?",
                "Anxiety often involves 'what if' thinking. Counter it by: Focusing on what you can control today, challenging catastrophic thoughts, using mindfulness apps like Headspace. What resonates with you?"
            ],
            'stress': [
                "Stress can be managed with specific strategies: 1) Prioritize your tasks and tackle one at a time, 2) Take regular 5-minute breaks, 3) Practice saying 'no' to non-essential commitments. What area needs attention first?",
                "When overwhelmed by stress, try: Time-blocking your schedule, delegating tasks where possible, doing brief meditation sessions. Which strategy could help reduce your stress load?",
                "Stress management involves both immediate relief and long-term strategies: Deep breathing for immediate calm, regular exercise for ongoing resilience, boundary-setting for prevention. What's your priority?"
            ]
        }
        
        # Check provider availability
        self.check_providers()
    
    def check_providers(self):
        """Check availability of LLM providers"""
        logger.info("Checking LLM provider availability...")
        
        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.provider_status['ollama'] = True
                logger.info("✓ Ollama is available")
            else:
                logger.warning("Ollama server responded but with error status")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama not available: {e}")
            self.provider_status['ollama'] = False
        
        # Check Groq (if API key is available)
        if self.groq_api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                # Simple test request
                test_payload = {
                    "messages": [{"role": "user", "content": "test"}],
                    "model": self.groq_model,
                    "max_tokens": 1
                }
                response = requests.post(self.groq_url, headers=headers, json=test_payload, timeout=10)
                if response.status_code in [200, 400]:  # 400 is ok for test
                    self.provider_status['groq'] = True
                    logger.info("✓ Groq is available")
                else:
                    logger.warning(f"Groq test failed with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Groq not available: {e}")
                self.provider_status['groq'] = False
        else:
            logger.info("Groq API key not configured")
            self.provider_status['groq'] = False
        
        logger.info(f"Provider status: {self.provider_status}")
    
    def generate_response(self, user_message: str, analysis: Dict[str, Any], 
                         chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate therapeutic response using available LLM providers with fallbacks
        
        Args:
            user_message: User's input message
            analysis: Text analysis results
            chat_history: Previous conversation context
            
        Returns:
            Dictionary containing response and metadata
        """
        
        # Handle crisis situations first
        crisis_level = analysis.get('crisis_classification', 'LOW')
        if crisis_level in ['HIGH', 'CRITICAL']:
            return self._handle_crisis_response(user_message, analysis, crisis_level)
        
        # Try LLM providers in order
        response = None
        
        # Try Ollama first
        if self.provider_status['ollama']:
            response = self._try_ollama(user_message, analysis, chat_history)
            if response:
                return response
        
        # Try Groq second
        if self.provider_status['groq']:
            response = self._try_groq(user_message, analysis, chat_history)
            if response:
                return response
        
        # Fallback to rule-based response
        return self._generate_fallback_response(user_message, analysis)
    
    def _try_ollama(self, user_message: str, analysis: Dict[str, Any], 
                   chat_history: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Try generating response using Ollama"""
        try:
            logger.info("Attempting Ollama response generation...")
            
            # Build context
            context = self._build_therapeutic_context(user_message, analysis, chat_history)
            
            # Ollama API call
            payload = {
                "model": "llama3.1:8b",  # Default model, can be configured
                "prompt": context,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 300
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                if content:
                    logger.info("✓ Ollama response generated successfully")
                    return {
                        'content': content,
                        'provider': 'ollama',
                        'model': payload['model'],
                        'technique': self._identify_technique(content),
                        'confidence': 0.8,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    logger.warning("Ollama returned empty response")
            else:
                logger.warning(f"Ollama request failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            self.provider_status['ollama'] = False  # Mark as unavailable
        except Exception as e:
            logger.error(f"Unexpected error with Ollama: {e}")
        
        return None
    
    def _try_groq(self, user_message: str, analysis: Dict[str, Any], 
                 chat_history: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Try generating response using Groq"""
        try:
            logger.info("Attempting Groq response generation...")
            
            if not self.groq_api_key:
                logger.warning("Groq API key not configured")
                return None
            
            # Build messages for OpenAI-compatible API
            messages = self._build_groq_messages(user_message, analysis, chat_history)
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": messages,
                "model": self.groq_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": 0.9
            }
            
            response = requests.post(
                self.groq_url,
                headers=headers,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                if content:
                    logger.info("✓ Groq response generated successfully")
                    return {
                        'content': content,
                        'provider': 'groq',
                        'model': payload['model'],
                        'technique': self._identify_technique(content),
                        'confidence': 0.7,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    logger.warning("Groq returned empty response")
            else:
                logger.warning(f"Groq request failed with status {response.status_code}")
                logger.warning(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq request failed: {e}")
            self.provider_status['groq'] = False  # Mark as unavailable
        except Exception as e:
            logger.error(f"Unexpected error with Groq: {e}")
        
        return None
    
    def _build_therapeutic_context(self, user_message: str, analysis: Dict[str, Any], 
                                  chat_history: List[Dict[str, Any]] = None) -> str:
        """Build therapeutic context for Ollama"""
        
        # Extract key analysis information
        emotion = analysis.get('dominant_emotion', 'neutral')
        sentiment = analysis.get('sentiment', 'neutral')
        risk_level = analysis.get('risk_level', 'LOW')
        topics = analysis.get('mental_health_topics', [])
        techniques = analysis.get('suggested_techniques', [])
        
        # Build context
        context = f"""You are a compassionate, professional mental health therapist. Respond to the user with empathy and therapeutic techniques.

Analysis of user's message:
- Emotion: {emotion}
- Sentiment: {sentiment}
- Risk level: {risk_level}
- Topics: {', '.join([t[0] for t in topics[:3]])}
- Suggested techniques: {', '.join(techniques)}

User's message: "{user_message}"

Provide a therapeutic response that:
1. Acknowledges their feelings with empathy
2. Uses appropriate therapeutic techniques
3. Is supportive but not giving medical advice
4. Encourages professional help if needed
5. Keeps the response concise (2-3 sentences)

Therapeutic response:"""
        
        return context
    
    def _build_groq_messages(self, user_message: str, analysis: Dict[str, Any], 
                           chat_history: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Build message array for Groq OpenAI-compatible API"""
        
        # System message
        emotion = analysis.get('dominant_emotion', 'neutral')
        risk_level = analysis.get('risk_level', 'LOW')
        techniques = analysis.get('suggested_techniques', [])
        
        system_message = f"""You are a compassionate mental health therapist. The user is expressing {emotion} emotions with {risk_level} risk level. Consider using these techniques: {', '.join(techniques)}. 

Respond with empathy and professional therapeutic guidance. Provide specific, actionable suggestions rather than just asking questions. Keep responses concise and supportive. Do not provide medical advice."""
        
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add recent chat history if available
        if chat_history:
            for msg in chat_history[-4:]:  # Last 4 messages for context
                role = "user" if msg.get('role') == 'user' else "assistant"
                content = msg.get('content', '')
                if content:
                    messages.append({"role": role, "content": content})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _handle_crisis_response(self, user_message: str, analysis: Dict[str, Any], 
                              crisis_level: str) -> Dict[str, Any]:
        """Handle high-risk crisis situations with immediate intervention guidance"""
        
        import random
        
        crisis_responses = self.crisis_responses.get(crisis_level, self.crisis_responses['MODERATE'])
        base_response = random.choice(crisis_responses)
        
        # Add specific resources based on detected indicators
        crisis_indicators = analysis.get('crisis_indicators', [])
        additional_resources = []
        
        if any('suicide' in indicator for indicator in crisis_indicators):
            additional_resources.append("National Suicide Prevention Lifeline: 988")
        
        if any('self_harm' in indicator for indicator in crisis_indicators):
            additional_resources.append("Crisis Text Line: Text HOME to 741741")
        
        # Combine response with resources
        full_response = base_response
        if additional_resources:
            full_response += "\n\nImmediate resources:\n" + "\n".join(f"• {resource}" for resource in additional_resources)
        
        return {
            'content': full_response,
            'provider': 'crisis_protocol',
            'technique': 'crisis_intervention',
            'confidence': 1.0,
            'crisis_level': crisis_level,
            'requires_immediate_attention': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_fallback_response(self, user_message: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rule-based therapeutic response when LLMs are unavailable"""
        
        emotion = analysis.get('dominant_emotion', 'neutral')
        topics = analysis.get('mental_health_topics', [])
        primary_topic = topics[0][0] if topics else 'general'
        suggested_techniques = analysis.get('suggested_techniques', ['supportive_counseling'])
        
        # Check for specific situations in user message
        message_lower = user_message.lower()
        
        # Identify specific situations
        if any(word in message_lower for word in ['broke up', 'breakup', 'break up', 'relationship ended']):
            situation = 'breakup'
        elif any(word in message_lower for word in ['depressed', 'depression', 'hopeless', 'worthless']):
            situation = 'depression'
        elif any(word in message_lower for word in ['anxious', 'anxiety', 'worried', 'nervous', 'panic']):
            situation = 'anxiety'
        elif any(word in message_lower for word in ['stressed', 'stress', 'overwhelmed', 'pressure']):
            situation = 'stress'
        else:
            situation = emotion
        
        # Select appropriate response
        response = self._select_fallback_response(situation, primary_topic, suggested_techniques[0] if suggested_techniques else 'supportive_counseling')
        
        return {
            'content': response,
            'provider': 'rule_based',
            'technique': suggested_techniques[0] if suggested_techniques else 'supportive_counseling',
            'confidence': 0.6,
            'fallback_reason': 'llm_unavailable',
            'timestamp': datetime.now().isoformat()
        }
    
    def _select_fallback_response(self, situation: str, topic: str, technique: str) -> str:
        """Select appropriate rule-based response with actionable suggestions"""
        
        import random
        
        # Check for actionable responses first
        if situation in self.actionable_responses:
            return random.choice(self.actionable_responses[situation])
        
        # Emotion-based responses with more specific guidance
        emotion_responses = {
            'sadness': [
                "I can hear the sadness in your words. Here are some gentle steps that might help: 1) Allow yourself to feel these emotions without judgment, 2) Reach out to someone you trust, 3) Engage in a small self-care activity. What feels most manageable right now?",
                "Sadness is a natural response to difficult situations. Consider: Writing in a journal for 10 minutes, taking a warm shower or bath, listening to music that soothes you. Which of these resonates with you?",
                "When we're feeling sad, small acts of self-compassion can help. Try: Speaking to yourself as you would a good friend, doing one thing that usually brings comfort, or simply resting without guilt. What sounds possible today?"
            ],
            'anger': [
                "I can sense the frustration and anger. Here are some ways to process these feelings: 1) Try physical movement like walking or stretching, 2) Write your thoughts down without editing, 3) Practice deep breathing for 2 minutes. What feels right for you?",
                "Anger often signals that something important needs attention. Consider: Identifying what boundary was crossed, expressing your feelings in a journal, or talking to someone you trust. Which approach appeals to you?",
                "When anger feels overwhelming, try: The 'STOP' technique (Stop, Take a breath, Observe, Proceed mindfully), physical exercise to release tension, or productive problem-solving. What would help most right now?"
            ],
            'fear': [
                "Fear can feel overwhelming, but you can take back some control. Try: Breaking down your worry into smaller, specific concerns, focusing on what you can influence today, or using the 5-4-3-2-1 grounding technique. What feels most helpful?",
                "When fear takes over, grounding can help: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste. This brings you back to the present moment.",
                "Fear often involves 'what if' thinking. Counter it by: Focusing on facts vs. fears, making a simple plan for what you can control, or practicing breathing exercises. Which strategy resonates with you?"
            ]
        }
        
        # Topic-based responses with actionable steps
        topic_responses = {
            'depression': self.actionable_responses['depression'],
            'anxiety': self.actionable_responses['anxiety'],
            'stress': self.actionable_responses['stress']
        }
        
        # Select response
        if situation in emotion_responses:
            return random.choice(emotion_responses[situation])
        elif topic in topic_responses:
            return random.choice(topic_responses[topic])
        else:
            # Generic supportive responses with action items
            generic_responses = [
                "Thank you for sharing this with me. It takes courage to open up. Here are some things that might help: 1) Practice self-compassion, 2) Connect with supportive people, 3) Focus on small, manageable goals. What feels most important to you right now?",
                "I hear you, and your feelings are completely valid. Consider these steps: Taking things one day at a time, identifying one person you can talk to, or doing one small thing that brings you comfort. What sounds most helpful?",
                "You're going through a challenging time, and that's okay. Some things that often help: Maintaining basic self-care routines, reaching out for support, and being patient with yourself. Which area would you like to focus on first?"
            ]
            return random.choice(generic_responses)
    
    def _identify_technique(self, response_content: str) -> str:
        """Identify the therapeutic technique used in a response"""
        
        content_lower = response_content.lower()
        
        # Check for technique keywords
        technique_keywords = {
            'cognitive_restructuring': ['thought', 'thinking', 'perspective', 'evidence', 'challenge'],
            'behavioral_activation': ['activity', 'action', 'step', 'doing', 'behavior'],
            'mindfulness': ['present', 'moment', 'aware', 'notice', 'breathing', 'focus'],
            'grounding': ['ground', 'safe', 'here', 'now', 'calm', 'breathe'],
            'validation': ['understand', 'hear', 'valid', 'normal', 'okay'],
            'psychoeducation': ['depression', 'anxiety', 'common', 'treatable', 'condition']
        }
        
        for technique, keywords in technique_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return technique
        
        return 'supportive_counseling'
    
    def set_groq_api_key(self, api_key: str):
        """Set Groq API key and recheck availability"""
        self.groq_api_key = api_key
        self.check_providers()
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get current status of all providers"""
        return {
            'providers': self.provider_status,
            'primary': 'ollama' if self.provider_status['ollama'] else 'groq' if self.provider_status['groq'] else 'rule_based',
            'last_checked': datetime.now().isoformat()
        }
    
    def test_providers(self) -> Dict[str, Any]:
        """Test all providers and return detailed status"""
        test_message = "I'm feeling a bit anxious about work."
        test_analysis = {
            'dominant_emotion': 'anxiety',
            'sentiment': 'neutral',
            'risk_level': 'LOW',
            'crisis_classification': 'LOW',
            'mental_health_topics': [('anxiety', 0.7)],
            'suggested_techniques': ['mindfulness']
        }
        
        results = {}
        
        # Test Ollama
        if self.provider_status['ollama']:
            ollama_result = self._try_ollama(test_message, test_analysis)
            results['ollama'] = {
                'available': ollama_result is not None,
                'response_length': len(ollama_result['content']) if ollama_result else 0
            }
        else:
            results['ollama'] = {'available': False, 'reason': 'not_connected'}
        
        # Test Groq
        if self.provider_status['groq']:
            groq_result = self._try_groq(test_message, test_analysis)
            results['groq'] = {
                'available': groq_result is not None,
                'response_length': len(groq_result['content']) if groq_result else 0
            }
        else:
            results['groq'] = {'available': False, 'reason': 'api_key_missing'}
        
        # Test fallback
        fallback_result = self._generate_fallback_response(test_message, test_analysis)
        results['fallback'] = {
            'available': True,
            'response_length': len(fallback_result['content'])
        }
        
        return results


# Test function
def test_therapy_agent():
    """Test the therapy agent with sample scenarios"""
    agent = TherapyAgent()
    
    test_cases = [
        {
            'message': "I'm feeling really anxious about work",
            'analysis': {
                'dominant_emotion': 'anxiety',
                'sentiment': 'negative',
                'risk_level': 'LOW',
                'crisis_classification': 'LOW',
                'mental_health_topics': [('anxiety', 0.8)],
                'suggested_techniques': ['mindfulness', 'relaxation']
            }
        },
        {
            'message': "I can't take this anymore",
            'analysis': {
                'dominant_emotion': 'sadness',
                'sentiment': 'negative',
                'risk_level': 'HIGH',
                'crisis_classification': 'HIGH',
                'crisis_indicators': ['desperation: "can\'t take this anymore"'],
                'mental_health_topics': [('depression', 0.9)],
                'suggested_techniques': ['crisis_intervention']
            }
        }
    ]
    
    print("Testing Therapy Agent")
    print("=" * 50)
    
    # Check provider status
    status = agent.get_provider_status()
    print(f"Provider Status: {status}")
    print()
    
    # Test responses
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['message']}")
        response = agent.generate_response(
            test_case['message'],
            test_case['analysis']
        )
        
        print(f"Provider: {response['provider']}")
        print(f"Technique: {response['technique']}")
        print(f"Response: {response['content'][:100]}...")
        if response.get('requires_immediate_attention'):
            print("🚨 CRISIS RESPONSE TRIGGERED")
        print("-" * 30)


if __name__ == "__main__":
    test_therapy_agent()