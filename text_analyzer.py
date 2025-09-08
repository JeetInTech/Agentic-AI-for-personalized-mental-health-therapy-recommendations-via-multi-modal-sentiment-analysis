"""
Simplified Text Analyzer for Mental Health Therapy System
Phase 1: Text-only analysis with reliable models and robust error handling
"""

import torch
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    """
    Simplified, robust text analyzer for mental health applications
    Uses only verified working HuggingFace models with comprehensive fallbacks
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Analysis pipelines - will be loaded safely
        self.emotion_pipeline = None
        self.sentiment_pipeline = None
        
        # Working model names (verified to exist)
        self.models = {
            'emotion': "j-hartmann/emotion-english-distilroberta-base",
            'sentiment': "cardiffnlp/twitter-roberta-base-sentiment-latest"
        }
        
        # Crisis detection keywords
        self.crisis_keywords = {
            'suicide': ['suicide', 'kill myself', 'end my life', 'want to die', 'better off dead', 'end it all'],
            'self_harm': ['cut myself', 'hurt myself', 'self harm', 'self-harm', 'cutting', 'burning myself'],
            'hopelessness': ['hopeless', 'no point', 'give up', 'nothing matters', 'no way out', 'no hope'],
            'desperation': ['desperate', "can't go on", 'unbearable', 'too much pain', "can't take it"]
        }
        
        # Mental health topics
        self.mental_health_topics = {
            'depression': ['depressed', 'sad', 'empty', 'worthless', 'hopeless', 'numb', 'down'],
            'anxiety': ['anxious', 'worried', 'panic', 'fear', 'nervous', 'stressed', 'overwhelmed'],
            'trauma': ['trauma', 'ptsd', 'flashback', 'triggered', 'abuse', 'assault'],
            'grief': ['grief', 'loss', 'death', 'mourning', 'bereaved', 'passed away'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'burnout', 'exhausted'],
            'relationships': ['lonely', 'rejected', 'abandoned', 'betrayed', 'isolated'],
            'work': ['job stress', 'work pressure', 'unemployed', 'career', 'fired']
        }
        
        # Load models with error handling
        self.load_models()
    
    def load_models(self):
        """Load HuggingFace models with comprehensive error handling"""
        logger.info("Loading HuggingFace models...")
        
        # Load emotion model
        try:
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.models['emotion'],
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            logger.info("✓ Emotion model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self.emotion_pipeline = None
        
        # Load sentiment model
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.models['sentiment'],
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✓ Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self.sentiment_pipeline = None
        
        # Check if any models loaded
        if self.emotion_pipeline is None and self.sentiment_pipeline is None:
            logger.warning("No models loaded successfully - using fallback analysis only")
        else:
            logger.info("Model loading completed")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Main text analysis function with robust error handling
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if not text or not text.strip():
            return self._empty_result()
        
        text = text.strip()
        
        result = {
            'input_text': text,
            'timestamp': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'char_count': len(text),
            'analysis_method': 'hybrid'
        }
        
        try:
            # Core analyses with error handling
            result.update(self._analyze_emotions(text))
            result.update(self._analyze_sentiment(text))
            result.update(self._detect_crisis_indicators(text))
            result.update(self._detect_mental_health_topics(text))
            result.update(self._analyze_linguistic_features(text))
            
            # Compute overall assessment
            result.update(self._compute_overall_assessment(result))
            
            logger.info(f"Analysis completed for text: {text[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return self._error_result(text, str(e))
    
    def _analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze emotions with robust error handling"""
        try:
            if self.emotion_pipeline is None:
                return self._fallback_emotion_analysis(text)
            
            # Get emotion predictions with error handling
            results = self.emotion_pipeline(text)
            
            if not results or not isinstance(results, list) or len(results) == 0:
                logger.warning("Emotion pipeline returned empty results")
                return self._fallback_emotion_analysis(text)
            
            # Process results safely
            emotion_scores = {}
            emotion_list = results[0] if isinstance(results[0], list) else results
            
            for emotion_data in emotion_list:
                if isinstance(emotion_data, dict) and 'label' in emotion_data and 'score' in emotion_data:
                    label = str(emotion_data['label']).lower()
                    
                    # Safely convert score to float
                    try:
                        score = float(emotion_data['score'])
                        emotion_scores[label] = score
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert emotion score to float: {emotion_data['score']}")
                        continue
                else:
                    logger.warning(f"Unexpected emotion data format: {emotion_data}")
                    continue
            
            if not emotion_scores:
                logger.warning("No valid emotion scores extracted")
                return self._fallback_emotion_analysis(text)
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            return {
                'dominant_emotion': dominant_emotion[0],
                'emotion_confidence': float(dominant_emotion[1]),
                'emotion_scores': emotion_scores,
                'emotion_method': 'huggingface'
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return self._fallback_emotion_analysis(text)
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment with robust error handling"""
        try:
            if self.sentiment_pipeline is None:
                return self._fallback_sentiment_analysis(text)
            
            # Get sentiment prediction with error handling
            results = self.sentiment_pipeline(text)
            
            if not results or not isinstance(results, list) or len(results) == 0:
                logger.warning("Sentiment pipeline returned empty results")
                return self._fallback_sentiment_analysis(text)
            
            sentiment_result = results[0]
            
            if not isinstance(sentiment_result, dict) or 'label' not in sentiment_result or 'score' not in sentiment_result:
                logger.warning(f"Unexpected sentiment result format: {sentiment_result}")
                return self._fallback_sentiment_analysis(text)
            
            # Map labels to standard format with error handling
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            raw_label = str(sentiment_result['label']).upper()
            sentiment_label = label_mapping.get(raw_label, raw_label.lower())
            
            # Safely convert score to float
            try:
                sentiment_score = float(sentiment_result['score'])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert sentiment score to float: {sentiment_result['score']}")
                return self._fallback_sentiment_analysis(text)
            
            return {
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_score,
                'sentiment_method': 'huggingface'
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _detect_crisis_indicators(self, text: str) -> Dict[str, Any]:
        """Detect crisis indicators using keyword-based approach"""
        text_lower = text.lower()
        crisis_score = 0.0
        detected_indicators = []
        
        # Check for crisis keywords
        for category, keywords in self.crisis_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category == 'suicide':
                        crisis_score += 0.4
                    elif category == 'self_harm':
                        crisis_score += 0.3
                    elif category == 'hopelessness':
                        crisis_score += 0.2
                    else:
                        crisis_score += 0.15
                    detected_indicators.append(f"{category}: '{keyword}'")
        
        # Check for additional high-risk patterns
        high_risk_patterns = [
            'want to die', 'kill myself', 'end it all', 'better off dead',
            'no reason to live', 'can\'t go on', 'too much pain'
        ]
        
        for pattern in high_risk_patterns:
            if pattern in text_lower:
                crisis_score += 0.3
                if f"suicide: '{pattern}'" not in detected_indicators:
                    detected_indicators.append(f"high_risk: '{pattern}'")
        
        # Check for isolation/hopelessness indicators
        isolation_patterns = ['nobody cares', 'all alone', 'no one understands', 'completely alone']
        for pattern in isolation_patterns:
            if pattern in text_lower:
                crisis_score += 0.1
                detected_indicators.append(f"isolation: '{pattern}'")
        
        # Normalize crisis score
        crisis_score = min(crisis_score, 1.0)
        
        # Determine crisis level
        if crisis_score >= 0.8:
            crisis_level = "CRITICAL"
        elif crisis_score >= 0.6:
            crisis_level = "HIGH"
        elif crisis_score >= 0.3:
            crisis_level = "MODERATE"
        else:
            crisis_level = "LOW"
        
        return {
            'crisis_level': float(crisis_score),
            'crisis_classification': crisis_level,
            'crisis_indicators': detected_indicators,
            'requires_intervention': crisis_score >= 0.5
        }
    
    def _detect_mental_health_topics(self, text: str) -> Dict[str, Any]:
        """Detect mental health topics mentioned in text"""
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in self.mental_health_topics.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                confidence = min(matches / len(keywords) * 2, 1.0)
                detected_topics.append((topic, confidence))
        
        # Sort by confidence
        detected_topics.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'mental_health_topics': detected_topics,
            'primary_topic': detected_topics[0][0] if detected_topics else 'general',
            'topic_confidence': detected_topics[0][1] if detected_topics else 0.0
        }
    
    def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns that may indicate mental state"""
        words = text.split()
        
        if not words:
            return {
                'first_person_ratio': 0.0,
                'negative_language_ratio': 0.0,
                'absolutist_thinking_ratio': 0.0,
                'sentence_count': 0,
                'avg_word_length': 0.0
            }
        
        # First person pronoun usage (self-focus indicator)
        first_person_words = ['i', 'me', 'my', 'myself', 'mine']
        first_person_count = sum(1 for word in words if word.lower() in first_person_words)
        first_person_ratio = first_person_count / len(words)
        
        # Negative language
        negative_words = ['not', 'no', 'never', 'nothing', 'nobody', 'can\'t', 'won\'t', 'don\'t']
        negative_count = sum(1 for word in words if word.lower() in negative_words)
        negative_ratio = negative_count / len(words)
        
        # Absolutist thinking
        absolutist_words = ['always', 'never', 'everything', 'nothing', 'everyone', 'nobody', 'all', 'none']
        absolutist_count = sum(1 for word in words if word.lower() in absolutist_words)
        absolutist_ratio = absolutist_count / len(words)
        
        # Sentence analysis
        sentences = [s for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        return {
            'first_person_ratio': float(first_person_ratio),
            'negative_language_ratio': float(negative_ratio),
            'absolutist_thinking_ratio': float(absolutist_ratio),
            'sentence_count': sentence_count,
            'avg_word_length': float(avg_word_length)
        }
    
    def _compute_overall_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall risk and therapeutic recommendations"""
        
        # Risk calculation with weights
        crisis_weight = 0.5
        sentiment_weight = 0.3
        emotion_weight = 0.2
        
        # Crisis component
        crisis_component = analysis.get('crisis_level', 0.0)
        
        # Sentiment component (negative sentiment increases risk)
        sentiment = analysis.get('sentiment', 'neutral')
        sentiment_score = analysis.get('sentiment_score', 0.5)
        if sentiment == 'negative':
            sentiment_component = sentiment_score
        elif sentiment == 'positive':
            sentiment_component = 1.0 - sentiment_score
        else:
            sentiment_component = 0.3
        
        # Emotion component (negative emotions increase risk)
        emotion = analysis.get('dominant_emotion', 'neutral')
        negative_emotions = ['sadness', 'fear', 'anger', 'disgust', 'disappointment']
        if emotion in negative_emotions:
            emotion_component = analysis.get('emotion_confidence', 0.5)
        else:
            emotion_component = 0.2
        
        # Overall risk score
        overall_risk = (
            crisis_component * crisis_weight +
            sentiment_component * sentiment_weight +
            emotion_component * emotion_weight
        )
        
        # Risk classification
        if overall_risk >= 0.8:
            risk_level = "HIGH"
            suggested_action = "immediate_intervention"
        elif overall_risk >= 0.5:
            risk_level = "MODERATE"
            suggested_action = "professional_support"
        elif overall_risk >= 0.3:
            risk_level = "LOW_MODERATE"
            suggested_action = "monitoring"
        else:
            risk_level = "LOW"
            suggested_action = "general_support"
        
        # Therapeutic technique suggestions
        suggested_techniques = self._suggest_techniques(analysis)
        
        return {
            'overall_risk_score': float(overall_risk),
            'risk_level': risk_level,
            'suggested_action': suggested_action,
            'suggested_techniques': suggested_techniques,
            'analysis_confidence': self._calculate_confidence(analysis)
        }
    
    def _suggest_techniques(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest therapeutic techniques based on analysis"""
        techniques = []
        
        # Based on emotion
        emotion = analysis.get('dominant_emotion', 'neutral')
        emotion_techniques = {
            'sadness': ['Behavioral Activation', 'Cognitive Restructuring'],
            'anger': ['Anger Management', 'Mindfulness'],
            'fear': ['Exposure Therapy', 'Grounding Techniques'],
            'anxiety': ['Progressive Muscle Relaxation', 'Breathing Exercises'],
            'joy': ['Positive Psychology', 'Gratitude Practice']
        }
        techniques.extend(emotion_techniques.get(emotion, []))
        
        # Based on mental health topic
        topic = analysis.get('primary_topic', 'general')
        topic_techniques = {
            'depression': ['CBT', 'Behavioral Activation'],
            'anxiety': ['Mindfulness', 'Relaxation Techniques'],
            'trauma': ['EMDR', 'Grounding Techniques'],
            'stress': ['Stress Management', 'Time Management'],
            'relationships': ['Communication Skills', 'Boundary Setting']
        }
        techniques.extend(topic_techniques.get(topic, []))
        
        # Based on linguistic patterns
        if analysis.get('absolutist_thinking_ratio', 0) > 0.1:
            techniques.append('Cognitive Restructuring')
        
        if analysis.get('first_person_ratio', 0) > 0.2:
            techniques.append('Mindfulness-Based Therapy')
        
        # Remove duplicates and return top 3
        unique_techniques = list(set(techniques))
        return unique_techniques[:3] if unique_techniques else ['Supportive Counseling']
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence in analysis"""
        factors = []
        
        # Text length factor
        word_count = analysis.get('word_count', 0)
        length_factor = min(word_count / 20, 1.0)  # Max confidence at 20+ words
        factors.append(length_factor)
        
        # Model confidence factors
        factors.append(analysis.get('emotion_confidence', 0.5))
        factors.append(analysis.get('sentiment_confidence', 0.5))
        factors.append(analysis.get('topic_confidence', 0.5))
        
        return float(np.mean(factors))
    
    def _fallback_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback emotion analysis using keywords"""
        emotion_keywords = {
            'joy': ['happy', 'excited', 'joyful', 'glad', 'cheerful', 'pleased'],
            'sadness': ['sad', 'crying', 'tears', 'depressed', 'down', 'miserable'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
            'disgust': ['disgusted', 'revolted', 'sick', 'appalled']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = min(score / len(keywords), 1.0)
        
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return {
                'dominant_emotion': dominant_emotion[0],
                'emotion_confidence': float(dominant_emotion[1]),
                'emotion_scores': emotion_scores,
                'emotion_method': 'keyword_fallback'
            }
        else:
            return {
                'dominant_emotion': 'neutral',
                'emotion_confidence': 0.5,
                'emotion_scores': {'neutral': 1.0},
                'emotion_method': 'keyword_fallback'
            }
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using keywords"""
        positive_words = ['good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'sad', 'hate', 'awful', 'horrible', 'worst']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if neg_count > pos_count:
            return {
                'sentiment': 'negative',
                'sentiment_score': 0.7,
                'sentiment_confidence': 0.6,
                'sentiment_method': 'keyword_fallback'
            }
        elif pos_count > neg_count:
            return {
                'sentiment': 'positive',
                'sentiment_score': 0.7,
                'sentiment_confidence': 0.6,
                'sentiment_method': 'keyword_fallback'
            }
        else:
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.5,
                'sentiment_confidence': 0.5,
                'sentiment_method': 'keyword_fallback'
            }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return result for empty input"""
        return {
            'input_text': '',
            'timestamp': datetime.now().isoformat(),
            'word_count': 0,
            'char_count': 0,
            'dominant_emotion': 'neutral',
            'emotion_confidence': 0.0,
            'emotion_scores': {'neutral': 1.0},
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
            'analysis_confidence': 0.0
        }
    
    def _error_result(self, text: str, error: str) -> Dict[str, Any]:
        """Return result for analysis errors"""
        result = self._empty_result()
        result.update({
            'input_text': text,
            'word_count': len(text.split()) if text else 0,
            'char_count': len(text) if text else 0,
            'error': error,
            'analysis_confidence': 0.2,
            'analysis_method': 'error_fallback'
        })
        return result


# Test function
def test_text_analyzer():
    """Test the text analyzer with sample inputs"""
    analyzer = TextAnalyzer()
    
    test_cases = [
        "I'm feeling really happy today!",
        "I'm so sad and don't know what to do",
        "Work has been really stressful lately",
        "I can't take this anymore, I want to end it all",
        "I feel anxious about the presentation tomorrow"
    ]
    
    print("Testing Simplified Text Analyzer")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        result = analyzer.analyze_text(text)
        
        print(f"Emotion: {result['dominant_emotion']} ({result['emotion_confidence']:.2f})")
        print(f"Sentiment: {result['sentiment']} ({result['sentiment_score']:.2f})")
        print(f"Crisis Level: {result['crisis_level']:.2f} ({result['crisis_classification']})")
        print(f"Risk: {result['risk_level']} ({result['overall_risk_score']:.2f})")
        print(f"Techniques: {', '.join(result['suggested_techniques'])}")
        print(f"Method: {result.get('emotion_method', 'unknown')}")
        print("-" * 30)


if __name__ == "__main__":
    test_text_analyzer()