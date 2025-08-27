"""
Agentic AI Therapy Assistant - Autonomous therapeutic intervention system
Provides personalized mental health support through evidence-based therapeutic techniques
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import random
from pathlib import Path
from dataclasses import dataclass
import re
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TherapeuticIntervention:
    """Structure for therapeutic interventions"""
    technique: str
    category: str
    description: str
    steps: List[str]
    duration_minutes: int
    effectiveness_score: float
    crisis_appropriate: bool
    personalization_factors: List[str]


@dataclass
class UserProfile:
    """User personalization profile"""
    preferred_techniques: List[str]
    effective_interventions: Dict[str, float]
    crisis_patterns: List[Dict]
    emotional_triggers: List[str]
    response_patterns: Dict[str, Any]
    therapy_goals: List[str]
    session_history: List[Dict]


class TherapyAgent:
    """
    Autonomous AI Therapy Agent that provides:
    - Personalized therapeutic interventions
    - Crisis detection and response
    - Adaptive learning from user interactions
    - Evidence-based technique selection
    """

    def __init__(self):
        self.initialize_therapeutic_knowledge()
        self.setup_crisis_protocols()
        self.initialize_personalization_engine()
        self.load_user_profiles()
        self.setup_response_templates()

    def initialize_therapeutic_knowledge(self):
        """Initialize evidence-based therapeutic techniques database"""

        # CBT (Cognitive Behavioral Therapy) Techniques
        self.cbt_techniques = {
            'cognitive_restructuring': TherapeuticIntervention(
                technique='Cognitive Restructuring',
                category='CBT',
                description='Challenge and reframe negative thought patterns',
                steps=[
                    "Let's examine this thought: is it based on facts or feelings?",
                    "What evidence supports this thought? What evidence contradicts it?",
                    "If a friend had this thought, what would you tell them?",
                    "What's a more balanced way to think about this situation?",
                    "How does this new perspective make you feel?"
                ],
                duration_minutes=10,
                effectiveness_score=0.85,
                crisis_appropriate=True,
                personalization_factors=['negative_thinking', 'catastrophizing', 'all_or_nothing']
            ),

            'thought_stopping': TherapeuticIntervention(
                technique='Thought Stopping',
                category='CBT',
                description='Interrupt negative thought spirals',
                steps=[
                    "I notice you're caught in a negative thought cycle. Let's interrupt it.",
                    "Take a deep breath and mentally say 'STOP' to these thoughts.",
                    "Now, shift your attention to 5 things you can see around you.",
                    "Name 4 things you can touch, 3 things you can hear.",
                    "Take three more deep breaths and notice how you feel now."
                ],
                duration_minutes=5,
                effectiveness_score=0.75,
                crisis_appropriate=True,
                personalization_factors=['rumination', 'anxiety', 'obsessive_thoughts']
            ),

            'behavioral_activation': TherapeuticIntervention(
                technique='Behavioral Activation',
                category='CBT',
                description='Increase positive activities to improve mood',
                steps=[
                    "Let's identify one small, positive activity you could do today.",
                    "It could be as simple as taking a short walk or listening to music.",
                    "What activity sounds manageable and slightly enjoyable?",
                    "When could you do this activity? Let's set a specific time.",
                    "Remember: the goal is action, not perfection."
                ],
                duration_minutes=8,
                effectiveness_score=0.8,
                crisis_appropriate=False,
                personalization_factors=['depression', 'low_motivation', 'isolation']
            )
        }

        # DBT (Dialectical Behavior Therapy) Techniques
        self.dbt_techniques = {
            'distress_tolerance': TherapeuticIntervention(
                technique='TIPP (Temperature, Intense Exercise, Paced Breathing, Paired Muscle Relaxation)',
                category='DBT',
                description='Rapidly reduce intense emotional distress',
                steps=[
                    "Let's use TIPP to manage this intense emotion quickly.",
                    "Temperature: Hold ice cubes or splash cold water on your face.",
                    "Or do 30 seconds of jumping jacks to change your body chemistry.",
                    "Now, let's do paced breathing: breathe in for 4, hold for 7, out for 8.",
                    "Tense your muscles for 5 seconds, then completely relax them."
                ],
                duration_minutes=5,
                effectiveness_score=0.9,
                crisis_appropriate=True,
                personalization_factors=['high_intensity_emotions', 'panic', 'rage']
            ),

            'wise_mind': TherapeuticIntervention(
                technique='Wise Mind',
                category='DBT',
                description='Balance emotional and rational thinking',
                steps=[
                    "Right now, what is your emotional mind telling you?",
                    "What is your rational mind saying about this situation?",
                    "Let's find your wise mind - the intersection of both.",
                    "Take a moment to breathe and ask: what would be most helpful right now?",
                    "What action honors both your feelings and your wisdom?"
                ],
                duration_minutes=7,
                effectiveness_score=0.8,
                crisis_appropriate=True,
                personalization_factors=['emotional_overwhelm', 'decision_making', 'conflict']
            ),

            'radical_acceptance': TherapeuticIntervention(
                technique='Radical Acceptance',
                category='DBT',
                description='Accept reality without approving of it',
                steps=[
                    "This situation is causing you pain. Let's practice radical acceptance.",
                    "Repeat to yourself: 'This is the reality right now.'",
                    "Notice any resistance - that's normal. Breathe through it.",
                    "Acceptance doesn't mean approval. You can accept reality and still work to change it.",
                    "How does it feel to stop fighting reality, even for a moment?"
                ],
                duration_minutes=10,
                effectiveness_score=0.75,
                crisis_appropriate=True,
                personalization_factors=['trauma', 'grief', 'chronic_pain', 'unchangeable_situations']
            )
        }

        # ACT (Acceptance and Commitment Therapy) Techniques
        self.act_techniques = {
            'values_clarification': TherapeuticIntervention(
                technique='Values Clarification',
                category='ACT',
                description='Connect with personal values for meaningful action',
                steps=[
                    "Let's step back from this problem and connect with your values.",
                    "What matters most to you in life? Family, creativity, helping others?",
                    "If you could live according to these values, what would that look like?",
                    "How can we take one small step toward these values today?",
                    "Remember: you can choose values-based action even when feeling difficult emotions."
                ],
                duration_minutes=12,
                effectiveness_score=0.85,
                crisis_appropriate=False,
                personalization_factors=['lack_of_direction', 'meaninglessness', 'major_life_changes']
            ),

            'cognitive_defusion': TherapeuticIntervention(
                technique='Cognitive Defusion',
                category='ACT',
                description='Create distance from unhelpful thoughts',
                steps=[
                    "I notice you're having the thought: [repeat their thought]",
                    "Now try saying: 'I'm having the thought that...' before the thought.",
                    "Now try: 'I notice I'm having the thought that...'",
                    "Sing the thought to the tune of 'Happy Birthday' - notice what happens.",
                    "These are just thoughts, not facts. How does this distance feel?"
                ],
                duration_minutes=6,
                effectiveness_score=0.8,
                crisis_appropriate=True,
                personalization_factors=['thought_fusion', 'self_criticism', 'limiting_beliefs']
            )
        }

        # Mindfulness and Grounding Techniques
        self.mindfulness_techniques = {
            '5_4_3_2_1_grounding': TherapeuticIntervention(
                technique='5-4-3-2-1 Grounding',
                category='Mindfulness',
                description='Ground yourself in the present moment using senses',
                steps=[
                    "Let's ground you in the present moment using your senses.",
                    "Name 5 things you can see around you right now.",
                    "Now 4 things you can touch or feel (your chair, your clothes, temperature).",
                    "3 things you can hear (maybe distant sounds, your breathing).",
                    "2 things you can smell, and 1 thing you can taste.",
                    "Notice how you feel more present and grounded now."
                ],
                duration_minutes=5,
                effectiveness_score=0.9,
                crisis_appropriate=True,
                personalization_factors=['anxiety', 'panic', 'dissociation', 'overwhelm']
            ),

            'breathing_space': TherapeuticIntervention(
                technique='Three-Minute Breathing Space',
                category='Mindfulness',
                description='Create space between you and difficult experiences',
                steps=[
                    "Let's take a three-minute breathing space together.",
                    "Minute 1: Awareness - What's happening right now? Notice thoughts, feelings, sensations.",
                    "Minute 2: Gathering - Bring your attention to your breath. Feel each inhale and exhale.",
                    "Minute 3: Expanding - Widen your awareness to your whole body and surroundings.",
                    "How do you feel after creating this space?"
                ],
                duration_minutes=3,
                effectiveness_score=0.85,
                crisis_appropriate=True,
                personalization_factors=['stress', 'emotional_reactivity', 'mindfulness_practice']
            )
        }

        # Combine all techniques
        self.all_techniques = {
            **self.cbt_techniques,
            **self.dbt_techniques,
            **self.act_techniques,
            **self.mindfulness_techniques
        }

    def setup_crisis_protocols(self):
        """Setup crisis intervention protocols"""
        self.crisis_responses = {
            'immediate': {
                'priority': 'Get immediate help',
                'message': """I'm very concerned about what you're sharing. Your safety is the most important thing right now.

**Immediate Help:**
🆘 Emergency: 911 (US) or your local emergency number
🆘 Crisis Hotline: 988 (US Suicide & Crisis Lifeline) 
🆘 Text HOME to 741741 (Crisis Text Line)

Please reach out to one of these resources right now. You don't have to go through this alone.""",
                'techniques': ['distress_tolerance', '5_4_3_2_1_grounding'],
                'follow_up': 'crisis_follow_up'
            },

            'high': {
                'priority': 'Urgent intervention needed',
                'message': """I can hear that you're going through a really difficult time. Let's work together to help you feel safer right now.

**Support Resources:**
📞 Crisis Hotline: 988 (available 24/7)
📱 Crisis Text Line: Text HOME to 741741
🌐 Online chat: suicidepreventionlifeline.org

Would you like to try a grounding technique with me first?""",
                'techniques': ['distress_tolerance', 'wise_mind', '5_4_3_2_1_grounding'],
                'follow_up': 'high_risk_follow_up'
            },

            'moderate': {
                'priority': 'Increased support recommended',
                'message': """I notice you're struggling right now. That takes courage to share. Let's work on some techniques to help you feel more stable.

Remember: difficult emotions are temporary, even when they feel overwhelming.""",
                'techniques': ['cognitive_restructuring', 'breathing_space', 'radical_acceptance'],
                'follow_up': 'moderate_risk_follow_up'
            }
        }

    def initialize_personalization_engine(self):
        """Setup personalization and learning systems"""
        self.personalization_factors = {
            # Personality traits affecting technique selection
            'personality': {
                'analytical': ['cognitive_restructuring', 'values_clarification'],
                'emotional': ['wise_mind', 'radical_acceptance'],
                'action_oriented': ['behavioral_activation', 'distress_tolerance'],
                'reflective': ['breathing_space', 'cognitive_defusion']
            },

            # Problem-specific technique mapping
            'presenting_issues': {
                'anxiety': ['5_4_3_2_1_grounding', 'thought_stopping', 'breathing_space'],
                'depression': ['behavioral_activation', 'cognitive_restructuring', 'values_clarification'],
                'trauma': ['distress_tolerance', 'radical_acceptance', '5_4_3_2_1_grounding'],
                'relationships': ['wise_mind', 'values_clarification', 'cognitive_defusion'],
                'anger': ['distress_tolerance', 'wise_mind', 'breathing_space'],
                'grief': ['radical_acceptance', 'breathing_space', 'values_clarification']
            },

            # Learning preferences
            'learning_style': {
                'structured': ['cognitive_restructuring', 'behavioral_activation'],
                'experiential': ['5_4_3_2_1_grounding', 'distress_tolerance'],
                'reflective': ['values_clarification', 'breathing_space'],
                'practical': ['thought_stopping', 'behavioral_activation']
            }
        }

        # Effectiveness tracking
        self.technique_effectiveness = {}
        self.user_feedback_history = []

    def load_user_profiles(self):
        """Load existing user profiles for personalization"""
        self.user_profiles = {}
        profiles_dir = Path('multimodal_profiles')

        if profiles_dir.exists():
            for profile_file in profiles_dir.glob('*.json'):
                try:
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)
                        user_id = profile_file.stem
                        # Ensure keys exist to match UserProfile dataclass
                        self.user_profiles[user_id] = UserProfile(
                            preferred_techniques=profile_data.get('preferred_techniques', []),
                            effective_interventions=profile_data.get('effective_interventions', {}),
                            crisis_patterns=profile_data.get('crisis_patterns', []),
                            emotional_triggers=profile_data.get('emotional_triggers', []),
                            response_patterns=profile_data.get('response_patterns', {}),
                            therapy_goals=profile_data.get('therapy_goals', []),
                            session_history=profile_data.get('session_history', [])
                        )
                except Exception as e:
                    print(f"Error loading profile {profile_file}: {e}")

    def setup_response_templates(self):
        """Setup response templates for different situations"""
        self.response_templates = {
            'empathy': [
                "I can hear that this is really difficult for you.",
                "It sounds like you're going through a lot right now.",
                "Thank you for sharing something so personal with me.",
                "I can sense the pain in what you're describing.",
                "It takes strength to reach out when you're struggling."
            ],

            'validation': [
                "Your feelings are completely valid.",
                "It makes sense that you would feel this way given what you've been through.",
                "Anyone in your situation would likely feel similarly.",
                "You're not overreacting - this is genuinely difficult.",
                "Your response is a normal reaction to an abnormal situation."
            ],

            'hope': [
                "While this feels overwhelming now, feelings do change over time.",
                "You've made it through difficult times before, and you can make it through this too.",
                "Taking this step to reach out shows your inner strength.",
                "Recovery and healing are possible, even when it doesn't feel that way.",
                "You don't have to face this alone."
            ],

            'transition': [
                "Let's try something that might help you feel a bit better right now.",
                "Would you be open to trying a technique with me?",
                "Let me guide you through something that often helps in situations like this.",
                "I'd like to teach you a skill that you can use whenever you need it.",
                "Let's work together on this."
            ]
        }

    def generate_response(self, user_message: str, analysis_results: Dict,
                          chat_history: List[Dict], user_id: str = 'default') -> Dict[str, Any]:
        """
        Generate personalized therapeutic response based on multimodal analysis

        Args:
            user_message: User's text input
            analysis_results: Results from multimodal fusion
            chat_history: Previous conversation history
            user_id: User identifier for personalization

        Returns:
            Dictionary containing response message, intervention, and metadata
        """

        # Assess situation severity
        crisis_risk = analysis_results.get('crisis_risk', 0)
        emotional_state = analysis_results.get('emotional_state', 'neutral')
        sentiment_score = analysis_results.get('sentiment_score', 0)
        confidence = analysis_results.get('overall_confidence', 0.5)

        # Get or create user profile
        user_profile = self.get_user_profile(user_id, analysis_results, chat_history)

        # Determine response strategy
        response_strategy = self.determine_response_strategy(
            crisis_risk, emotional_state, sentiment_score, user_profile
        )

        # Handle crisis situations first
        if crisis_risk > 0.6:
            return self.handle_crisis_response(crisis_risk, analysis_results, user_profile)

        # Generate therapeutic response
        response_components = self.build_therapeutic_response(
            user_message, analysis_results, user_profile, response_strategy
        )

        # Select and personalize intervention
        intervention = self.select_intervention(analysis_results, user_profile)

        # Update user profile based on interaction
        self.update_user_profile(user_id, analysis_results, intervention)

        return {
            'message': response_components['message'],
            'intervention': intervention,
            'strategy': response_strategy,
            'confidence': confidence,
            'personalization_applied': response_components['personalization_used'],
            'metadata': {
                'crisis_risk': crisis_risk,
                'techniques_considered': response_components['techniques_considered'],
                'timestamp': datetime.now().isoformat()
            }
        }

    def determine_response_strategy(self, crisis_risk: float, emotional_state: str,
                                    sentiment_score: float, user_profile: UserProfile) -> str:
        """Determine the appropriate response strategy"""

        if crisis_risk > 0.6:
            return 'crisis_intervention'
        elif sentiment_score < -0.7:
            return 'supportive_intervention'
        elif emotional_state in ['anxious', 'panic']:
            return 'anxiety_focused'
        elif emotional_state in ['sad', 'depressed']:
            return 'depression_focused'
        elif sentiment_score > 0.5:
            return 'positive_reinforcement'
        else:
            return 'exploratory_supportive'

    def handle_crisis_response(self, crisis_risk: float, analysis_results: Dict,
                               user_profile: UserProfile) -> Dict[str, Any]:
        """Handle crisis situations with appropriate escalation"""

        if crisis_risk > 0.8:
            crisis_level = 'immediate'
        elif crisis_risk > 0.7:
            crisis_level = 'high'
        else:
            crisis_level = 'moderate'

        crisis_protocol = self.crisis_responses[crisis_level]

        # Build crisis response message
        empathy_statement = random.choice(self.response_templates['empathy'])
        crisis_message = crisis_protocol['message']

        full_message = f"{empathy_statement}\n\n{crisis_message}"

        # Select crisis-appropriate technique
        available_techniques = [t for t in crisis_protocol['techniques']
                                if t in self.all_techniques]

        if available_techniques:
            # Personalize technique selection even in crisis
            preferred_technique = self.select_personalized_technique(
                available_techniques, user_profile, crisis_appropriate=True
            )
            technique_obj = self.all_techniques[preferred_technique]
        else:
            technique_obj = self.all_techniques['5_4_3_2_1_grounding']  # Fallback

        return {
            'message': full_message,
            'intervention': {
                'type': 'crisis_intervention',
                'priority': crisis_protocol['priority'],
                'technique': {
                    'name': technique_obj.technique,
                    'description': technique_obj.description,
                    'steps': technique_obj.steps
                },
                'follow_up': crisis_protocol['follow_up'],
                'crisis_level': crisis_level
            },
            'strategy': 'crisis_intervention',
            'confidence': 1.0,  # High confidence in crisis protocols
            'metadata': {
                'crisis_risk': crisis_risk,
                'crisis_indicators': analysis_results.get('crisis_indicators', []),
                'timestamp': datetime.now().isoformat()
            }
        }

    def build_therapeutic_response(self, user_message: str, analysis_results: Dict,
                                   user_profile: UserProfile, strategy: str) -> Dict[str, Any]:
        """Build comprehensive therapeutic response"""

        # Start with empathy and validation
        empathy_statement = self.select_empathy_statement(analysis_results, user_profile)
        validation_statement = self.select_validation_statement(analysis_results, user_profile)

        # Add therapeutic insight
        therapeutic_insight = self.generate_therapeutic_insight(
            analysis_results, user_profile, strategy
        )

        # Add hope and encouragement
        hope_statement = self.select_hope_statement(analysis_results, user_profile)

        # Transition to intervention
        transition_statement = random.choice(self.response_templates['transition'])

        # Combine components
        message_parts = [
            empathy_statement,
            validation_statement,
            therapeutic_insight,
            hope_statement,
            transition_statement
        ]

        # Remove empty parts and join
        message_parts = [part for part in message_parts if part]
        full_message = "\n\n".join(message_parts)

        return {
            'message': full_message,
            'personalization_used': self.get_personalization_applied(user_profile),
            'techniques_considered': list(self.all_techniques.keys())
        }

    def select_intervention(self, analysis_results: Dict, user_profile: UserProfile) -> Dict[str, Any]:
        """Select and personalize therapeutic intervention"""

        emotional_state = analysis_results.get('emotional_state', 'neutral')
        sentiment_score = analysis_results.get('sentiment_score', 0)
        crisis_risk = analysis_results.get('crisis_risk', 0)

        # Determine candidate techniques based on presenting issues
        candidate_techniques = []

        # Add techniques based on emotional state
        if emotional_state in ['anxious', 'panic']:
            candidate_techniques.extend(['5_4_3_2_1_grounding', 'breathing_space', 'distress_tolerance'])
        elif emotional_state in ['sad', 'depressed', 'very_negative']:
            candidate_techniques.extend(['cognitive_restructuring', 'behavioral_activation', 'values_clarification'])
        elif emotional_state in ['angry', 'frustrated']:
            candidate_techniques.extend(['distress_tolerance', 'wise_mind', 'radical_acceptance'])
        else:
            candidate_techniques.extend(['breathing_space', 'cognitive_defusion', 'wise_mind'])

        # Filter by crisis appropriateness
        if crisis_risk > 0.4:
            candidate_techniques = [t for t in candidate_techniques
                                    if self.all_techniques.get(t) and self.all_techniques[t].crisis_appropriate]

        # Personalize selection
        selected_technique = self.select_personalized_technique(
            candidate_techniques, user_profile, crisis_risk > 0.4
        )

        technique_obj = self.all_techniques[selected_technique]

        return {
            'type': 'therapeutic_technique',
            'technique': {
                'name': technique_obj.technique,
                'category': technique_obj.category,
                'description': technique_obj.description,
                'steps': technique_obj.steps,
                'duration_minutes': technique_obj.duration_minutes
            },
            'rationale': self.generate_technique_rationale(selected_technique, analysis_results),
            'personalization_factors': technique_obj.personalization_factors,
            'effectiveness_prediction': self.predict_technique_effectiveness(
                selected_technique, user_profile, analysis_results
            )
        }

    def select_personalized_technique(self, candidates: List[str], user_profile: UserProfile,
                                      crisis_appropriate: bool = False) -> str:
        """Select technique based on user preferences and history"""

        if not candidates:
            return '5_4_3_2_1_grounding'  # Safe fallback

        # Score techniques based on user preferences
        technique_scores = {}

        for technique in candidates:
            if technique not in self.all_techniques:
                continue

            score = 0.5  # Base score

            # Preference bonus
            if technique in user_profile.preferred_techniques:
                score += 0.3

            # Effectiveness history bonus
            if technique in user_profile.effective_interventions:
                score += user_profile.effective_interventions[technique] * 0.2

            # Crisis appropriateness
            if crisis_appropriate and self.all_techniques[technique].crisis_appropriate:
                score += 0.1

            # Technique effectiveness score
            score += self.all_techniques[technique].effectiveness_score * 0.2

            technique_scores[technique] = score

        # Select technique with highest score
        if technique_scores:
            return max(technique_scores.keys(), key=lambda x: technique_scores[x])
        else:
            return candidates[0]

    def generate_therapeutic_insight(self, analysis_results: Dict, user_profile: UserProfile,
                                     strategy: str) -> str:
        """Generate personalized therapeutic insight"""

        insights = analysis_results.get('insights', [])
        emotional_patterns = analysis_results.get('emotional_patterns', {})

        if insights:
            primary_insight = insights[0]

            # Personalize insight based on user's therapy history
            if 'declining' in primary_insight and 'CBT' in [pref.split('_')[0] for pref in user_profile.preferred_techniques]:
                return "I notice a pattern in your emotional experience. This might be a good time to examine the thoughts contributing to these feelings."
            elif 'anxiety' in primary_insight:
                return "Your body and mind are responding to stress right now. Let's work on bringing you back to a calmer state."
            elif 'positive' in primary_insight:
                return "I'm glad to hear some positive energy in what you're sharing. Let's build on this."
            else:
                return "I'm noticing some important patterns in what you're experiencing."

        return "Thank you for sharing what's on your mind."

    def select_empathy_statement(self, analysis_results: Dict, user_profile: UserProfile) -> str:
        """Select appropriate empathy statement"""
        sentiment_score = analysis_results.get('sentiment_score', 0)

        if sentiment_score < -0.6:
            return random.choice([
                "I can hear how much pain you're carrying right now.",
                "This sounds incredibly difficult and overwhelming.",
                "I can sense the heaviness in what you're sharing."
            ])
        elif sentiment_score < -0.2:
            return random.choice(self.response_templates['empathy'][:3])
        else:
            return random.choice(self.response_templates['empathy'][3:])

    def select_validation_statement(self, analysis_results: Dict, user_profile: UserProfile) -> str:
        """Select appropriate validation statement"""
        return random.choice(self.response_templates['validation'])

    def select_hope_statement(self, analysis_results: Dict, user_profile: UserProfile) -> str:
        """Select appropriate hope statement"""
        trajectory = analysis_results.get('emotional_trajectory', 'stable')

        if trajectory == 'declining':
            return "Even though things feel dark right now, you've shown resilience before and you have it within you now."
        elif trajectory == 'improving':
            return "I can sense some positive movement in how you're feeling. Let's build on that."
        else:
            return random.choice(self.response_templates['hope'])

    def generate_technique_rationale(self, technique: str, analysis_results: Dict) -> str:
        """Generate rationale for technique selection"""
        technique_obj = self.all_techniques[technique]
        emotional_state = analysis_results.get('emotional_state', 'neutral')

        rationales = {
            'cognitive_restructuring': f"This technique can help address the {emotional_state} thoughts that might be contributing to your distress.",
            'distress_tolerance': f"This technique is designed to help manage intense {emotional_state} feelings quickly and effectively.",
            '5_4_3_2_1_grounding': f"This grounding technique can help bring you back to the present moment when feeling {emotional_state}.",
            'behavioral_activation': "Engaging in positive activities can help improve mood and energy levels.",
            'wise_mind': "This technique helps balance emotional reactions with rational thinking."
        }

        return rationales.get(technique, f"This {technique_obj.category} technique is well-suited for your current emotional state.")

    def predict_technique_effectiveness(self, technique: str, user_profile: UserProfile,
                                        analysis_results: Dict) -> float:
        """Predict how effective this technique will be for this user"""

        base_effectiveness = self.all_techniques[technique].effectiveness_score

        # Adjust based on user history
        if technique in user_profile.effective_interventions:
            personal_effectiveness = user_profile.effective_interventions[technique]
            return (base_effectiveness + personal_effectiveness) / 2

        # Adjust based on user preferences
        preference_bonus = 0.1 if technique in user_profile.preferred_techniques else 0

        return min(base_effectiveness + preference_bonus, 1.0)

    def get_user_profile(self, user_id: str, analysis_results: Dict,
                         chat_history: List[Dict]) -> UserProfile:
        """Get or create user profile"""

        if user_id not in self.user_profiles:
            # Create new profile
            self.user_profiles[user_id] = UserProfile(
                preferred_techniques=[],
                effective_interventions={},
                crisis_patterns=[],
                emotional_triggers=[],
                response_patterns={},
                therapy_goals=[],
                session_history=[]
            )

        return self.user_profiles[user_id]

    def update_user_profile(self, user_id: str, analysis_results: Dict, intervention: Dict):
        """Update user profile based on interaction"""

        if user_id not in self.user_profiles:
            return

        profile = self.user_profiles[user_id]

        technique_name = intervention.get('technique', {}).get('name') or intervention.get('technique')
        timestamp = datetime.now().isoformat()

        session_entry = {
            'timestamp': timestamp,
            'emotional_state': analysis_results.get('emotional_state'),
            'sentiment_score': analysis_results.get('sentiment_score'),
            'intervention_used': technique_name,
            'crisis_risk': analysis_results.get('crisis_risk', 0),
            'crisis_indicators': analysis_results.get('crisis_indicators', []),
            'notes': analysis_results.get('notes', None)
        }

        profile.session_history.append(session_entry)

        observed_effectiveness = analysis_results.get('observed_effectiveness')
        if observed_effectiveness is None:
            feedback = analysis_results.get('user_feedback')  # optional: user self-report
            if isinstance(feedback, (int, float)):
                observed_effectiveness = float(feedback)

        if technique_name:
            prev = profile.effective_interventions.get(technique_name, None)
            if observed_effectiveness is not None:
                if prev is None:
                    profile.effective_interventions[technique_name] = float(observed_effectiveness)
                else:
                    profile.effective_interventions[technique_name] = float((prev + observed_effectiveness) / 2)

                global_prev = self.technique_effectiveness.get(technique_name, None)
                if global_prev is None:
                    self.technique_effectiveness[technique_name] = profile.effective_interventions[technique_name]
                else:
                    self.technique_effectiveness[technique_name] = (global_prev + profile.effective_interventions[technique_name]) / 2

        triggers = analysis_results.get('triggers', [])
        if isinstance(triggers, list):
            for t in triggers:
                if t not in profile.emotional_triggers:
                    profile.emotional_triggers.append(t)

        crisis_risk = analysis_results.get('crisis_risk', 0)
        if crisis_risk >= 0.6:
            pattern = {
                'timestamp': timestamp,
                'crisis_risk': crisis_risk,
                'indicators': analysis_results.get('crisis_indicators', []),
                'context_snippet': analysis_results.get('context_snippet')
            }
            profile.crisis_patterns.append(pattern)

        sentiment_history = profile.response_patterns.get('sentiment_history', [])
        sentiment_history.append({
            'timestamp': timestamp,
            'sentiment_score': analysis_results.get('sentiment_score', 0)
        })
        profile.response_patterns['sentiment_history'] = sentiment_history
        if len(sentiment_history) > 100:
            profile.response_patterns['sentiment_history'] = sentiment_history[-100:]

        self.save_user_profile(user_id)

    def save_user_profile(self, user_id: str):
        profiles_dir = Path('multimodal_profiles')
        profiles_dir.mkdir(parents=True, exist_ok=True)

        profile = self.user_profiles[user_id]

        serializable = {
            'preferred_techniques': list(profile.preferred_techniques),
            'effective_interventions': profile.effective_interventions,
            'crisis_patterns': profile.crisis_patterns,
            'emotional_triggers': profile.emotional_triggers,
            'response_patterns': profile.response_patterns,
            'therapy_goals': profile.therapy_goals,
            'session_history': []
        }

        for entry in profile.session_history:
            e = entry.copy()
            # If timestamp is a datetime, convert to isoformat
            if isinstance(e.get('timestamp'), datetime):
                e['timestamp'] = e['timestamp'].isoformat()
            serializable['session_history'].append(e)

        try:
            with open(profiles_dir / f"{user_id}.json", 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as ex:
            print(f"Error saving profile {user_id}: {ex}")

    def get_personalization_applied(self, user_profile: UserProfile) -> Dict[str, Any]:
        top_preferences = user_profile.preferred_techniques[:5]
        top_effective = sorted(user_profile.effective_interventions.items(), key=lambda x: -x[1])[:5]
        recent_sessions = user_profile.session_history[-5:]
        return {
            'top_preferences': top_preferences,
            'top_effective_interventions': top_effective,
            'recent_sessions_count': len(user_profile.session_history),
            'recent_sessions': recent_sessions
        }

    def adjust_technique_effectiveness(self, technique_name: str, observed_score: float):
        if not technique_name:
            return
        prev = self.technique_effectiveness.get(technique_name)
        if prev is None:
            self.technique_effectiveness[technique_name] = float(observed_score)
        else:
            self.technique_effectiveness[technique_name] = float((prev + observed_score) / 2)
