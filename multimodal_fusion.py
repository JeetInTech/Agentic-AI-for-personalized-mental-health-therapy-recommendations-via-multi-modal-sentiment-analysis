"""
Multimodal Fusion Engine - Combines text, audio, and visual analysis
Integrates sentiment analysis from multiple modalities for comprehensive emotional assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class MultimodalFusion:
    """
    Advanced multimodal fusion engine that combines analysis from:
    - Text: Sentiment, crisis keywords, linguistic patterns
    - Audio: Voice tone, prosody, vocal stress indicators
    - Visual: Facial expressions, micro-expressions, body language
    """
    
    def __init__(self):
        self.setup_fusion_parameters()
        self.initialize_models()
        self.setup_confidence_weighting()
        self.emotional_state_history = []
        self.pattern_cache = {}
        
    def setup_fusion_parameters(self):
        """Configure fusion parameters and weights"""
        # Modality importance weights (can be learned/adapted)
        self.modality_weights = {
            'text': 0.4,    # High weight for explicit communication
            'audio': 0.35,  # Voice tone often reveals hidden emotions
            'visual': 0.25  # Body language and micro-expressions
        }
        
        # Crisis detection weights (more conservative)
        self.crisis_weights = {
            'text': 0.5,    # Crisis keywords are highly indicative
            'audio': 0.3,   # Voice distress patterns
            'visual': 0.2   # Visual distress cues
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Emotional state mapping
        self.emotion_mapping = {
            'positive': ['happy', 'excited', 'content', 'grateful', 'optimistic'],
            'neutral': ['calm', 'focused', 'neutral', 'balanced'],
            'negative': ['sad', 'anxious', 'angry', 'frustrated', 'depressed'],
            'crisis': ['desperate', 'hopeless', 'suicidal', 'panic', 'breakdown']
        }
        
    def initialize_models(self):
        """Initialize fusion models and scalers"""
        self.scaler = MinMaxScaler()
        
        # Simple ensemble model for fusion (can be replaced with more complex models)
        self.fusion_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        # Pattern recognition parameters
        self.pattern_window = 5  # Number of previous states to consider
        self.stability_threshold = 0.15  # Threshold for emotional stability
        
    def setup_confidence_weighting(self):
        """Setup adaptive confidence weighting system"""
        self.confidence_factors = {
            'consistency': 0.3,  # How consistent across modalities
            'historical': 0.25,  # Consistency with user's history
            'certainty': 0.25,   # Individual modality confidence
            'context': 0.2       # Contextual appropriateness
        }
        
    def fuse_analyses(self, modality_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Main fusion function - combines all modality results
        
        Args:
            modality_results: Dictionary with keys 'text', 'audio', 'visual'
                            Each containing analysis results from respective analyzers
                            
        Returns:
            Fused analysis results with unified metrics
        """
        # Validate input
        if not modality_results:
            return self._default_analysis()
            
        available_modalities = list(modality_results.keys())
        
        # Extract core metrics from each modality
        extracted_metrics = self._extract_core_metrics(modality_results)
        
        # Compute unified sentiment
        unified_sentiment = self._compute_unified_sentiment(extracted_metrics)
        
        # Assess crisis risk
        crisis_assessment = self._assess_crisis_risk(extracted_metrics)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            modality_results, extracted_metrics
        )
        
        # Detect emotional patterns
        emotional_patterns = self._detect_emotional_patterns(unified_sentiment)
        
        # Generate insights
        insights = self._generate_insights(
            unified_sentiment, crisis_assessment, emotional_patterns, available_modalities
        )
        
        # Build comprehensive result
        fused_result = {
            'timestamp': datetime.now(),
            'modalities_analyzed': available_modalities,
            
            # Core sentiment metrics
            'sentiment_score': unified_sentiment['score'],
            'sentiment_label': unified_sentiment['label'],
            'sentiment_confidence': unified_sentiment['confidence'],
            
            # Crisis assessment
            'crisis_risk': crisis_assessment['risk_score'],
            'crisis_indicators': crisis_assessment['indicators'],
            'crisis_confidence': crisis_assessment['confidence'],
            
            # Confidence metrics
            'confidence_scores': confidence_scores,
            'overall_confidence': np.mean(list(confidence_scores.values())),
            
            # Emotional analysis
            'emotional_state': emotional_patterns['current_state'],
            'emotional_stability': emotional_patterns['stability'],
            'emotional_trajectory': emotional_patterns['trajectory'],
            
            # Modality breakdown
            'modality_contributions': self._calculate_modality_contributions(extracted_metrics),
            
            # Insights and recommendations
            'insights': insights,
            'recommendation_priority': self._calculate_recommendation_priority(crisis_assessment, emotional_patterns),
            
            # Technical details
            'fusion_metadata': {
                'weights_used': self.modality_weights,
                'pattern_window': self.pattern_window,
                'processing_time': datetime.now()
            }
        }
        
        # Update history
        self._update_emotional_history(fused_result)
        
        return fused_result
    
    def _extract_core_metrics(self, modality_results: Dict) -> Dict:
        """Extract standardized metrics from each modality"""
        extracted = {}
        
        # Text analysis extraction
        if 'text' in modality_results:
            text_data = modality_results['text']
            extracted['text'] = {
                'sentiment_score': text_data.get('sentiment_score', 0.0),
                'confidence': text_data.get('confidence', 0.5),
                'crisis_keywords': text_data.get('crisis_keywords', []),
                'emotions': text_data.get('emotions', {}),
                'intensity': text_data.get('intensity', 0.5)
            }
        
        # Audio analysis extraction
        if 'audio' in modality_results:
            audio_data = modality_results['audio']
            extracted['audio'] = {
                'sentiment_score': audio_data.get('emotional_valence', 0.0),
                'confidence': audio_data.get('confidence', 0.5),
                'stress_level': audio_data.get('stress_indicators', {}).get('overall_stress', 0.0),
                'voice_stability': audio_data.get('voice_stability', 0.5),
                'arousal_level': audio_data.get('arousal_level', 0.5)
            }
        
        # Visual analysis extraction
        if 'visual' in modality_results:
            visual_data = modality_results['visual']
            extracted['visual'] = {
                'sentiment_score': visual_data.get('overall_emotion_score', 0.0),
                'confidence': visual_data.get('confidence', 0.5),
                'facial_emotions': visual_data.get('facial_emotions', {}),
                'micro_expressions': visual_data.get('micro_expressions', []),
                'body_language_score': visual_data.get('body_language', {}).get('overall_score', 0.0)
            }
        
        return extracted
    
    def _compute_unified_sentiment(self, extracted_metrics: Dict) -> Dict:
        """Compute unified sentiment score across modalities"""
        sentiment_scores = []
        confidences = []
        modality_weights = []
        
        # Collect sentiment scores with weights
        for modality, data in extracted_metrics.items():
            if 'sentiment_score' in data:
                score = data['sentiment_score']
                confidence = data['confidence']
                weight = self.modality_weights.get(modality, 0.33)
                
                sentiment_scores.append(score)
                confidences.append(confidence)
                modality_weights.append(weight)
        
        if not sentiment_scores:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
        
        # Normalize weights
        total_weight = sum(modality_weights)
        normalized_weights = [w/total_weight for w in modality_weights]
        
        # Weighted average sentiment
        weighted_sentiment = sum(s * w for s, w in zip(sentiment_scores, normalized_weights))
        
        # Confidence-weighted average
        weighted_confidence = sum(c * w for c, w in zip(confidences, normalized_weights))
        
        # Apply consistency bonus/penalty
        consistency_factor = self._calculate_consistency_factor(sentiment_scores)
        final_confidence = weighted_confidence * consistency_factor
        
        # Determine sentiment label
        sentiment_label = self._score_to_label(weighted_sentiment)
        
        return {
            'score': weighted_sentiment,
            'label': sentiment_label,
            'confidence': final_confidence,
            'consistency_factor': consistency_factor,
            'modality_scores': dict(zip(extracted_metrics.keys(), sentiment_scores))
        }
    
    def _assess_crisis_risk(self, extracted_metrics: Dict) -> Dict:
        """Comprehensive crisis risk assessment"""
        crisis_indicators = []
        risk_scores = []
        
        # Text-based crisis indicators
        if 'text' in extracted_metrics:
            text_data = extracted_metrics['text']
            crisis_keywords = text_data.get('crisis_keywords', [])
            if crisis_keywords:
                crisis_indicators.extend([f"text: {kw}" for kw in crisis_keywords])
                # Crisis keyword severity scoring
                severity_map = {
                    'suicide': 0.9, 'kill': 0.85, 'die': 0.8, 'hopeless': 0.7,
                    'worthless': 0.65, 'hurt': 0.6, 'pain': 0.55, 'alone': 0.5
                }
                max_severity = max([severity_map.get(kw, 0.3) for kw in crisis_keywords] + [0])
                risk_scores.append(max_severity * self.crisis_weights['text'])
        
        # Audio-based crisis indicators
        if 'audio' in extracted_metrics:
            audio_data = extracted_metrics['audio']
            stress_level = audio_data.get('stress_level', 0)
            voice_stability = audio_data.get('voice_stability', 1)
            
            # High stress + low stability = crisis indicator
            if stress_level > 0.7 and voice_stability < 0.3:
                crisis_indicators.append("audio: high vocal distress")
                risk_scores.append(0.6 * self.crisis_weights['audio'])
            elif stress_level > 0.8:
                crisis_indicators.append("audio: extreme stress detected")
                risk_scores.append(0.7 * self.crisis_weights['audio'])
        
        # Visual-based crisis indicators
        if 'visual' in extracted_metrics:
            visual_data = extracted_metrics['visual']
            facial_emotions = visual_data.get('facial_emotions', {})
            micro_expressions = visual_data.get('micro_expressions', [])
            
            # Check for distress indicators
            distress_emotions = ['sadness', 'fear', 'disgust']
            high_distress = any(facial_emotions.get(emotion, 0) > 0.7 for emotion in distress_emotions)
            
            if high_distress:
                crisis_indicators.append("visual: high emotional distress")
                risk_scores.append(0.5 * self.crisis_weights['visual'])
            
            # Micro-expression analysis for concealed distress
            if any('distress' in expr for expr in micro_expressions):
                crisis_indicators.append("visual: concealed distress detected")
                risk_scores.append(0.4 * self.crisis_weights['visual'])
        
        # Calculate overall risk score
        overall_risk = sum(risk_scores) if risk_scores else 0.0
        overall_risk = min(overall_risk, 1.0)  # Cap at 1.0
        
        # Risk level determination
        if overall_risk > 0.8:
            risk_level = 'critical'
        elif overall_risk > 0.6:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'moderate'
        elif overall_risk > 0.2:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        return {
            'risk_score': overall_risk,
            'risk_level': risk_level,
            'indicators': crisis_indicators,
            'confidence': min(len(extracted_metrics) / 3.0, 1.0)  # More modalities = more confidence
        }
    
    def _calculate_confidence_scores(self, modality_results: Dict, extracted_metrics: Dict) -> Dict:
        """Calculate confidence scores for each modality"""
        confidence_scores = {}
        
        for modality in modality_results.keys():
            if modality in extracted_metrics:
                base_confidence = extracted_metrics[modality].get('confidence', 0.5)
                
                # Apply confidence factors
                consistency_bonus = self._get_consistency_bonus(modality, extracted_metrics)
                historical_bonus = self._get_historical_consistency_bonus(modality)
                context_bonus = self._get_context_appropriateness_bonus(modality, modality_results)
                
                # Calculate final confidence
                final_confidence = base_confidence * (1 + consistency_bonus + historical_bonus + context_bonus)
                final_confidence = min(final_confidence, 1.0)  # Cap at 1.0
                
                confidence_scores[modality] = final_confidence
        
        return confidence_scores
    
    def _detect_emotional_patterns(self, unified_sentiment: Dict) -> Dict:
        """Detect emotional patterns and trajectory"""
        current_score = unified_sentiment['score']
        current_label = unified_sentiment['label']
        
        # Add to history
        if len(self.emotional_state_history) >= self.pattern_window:
            self.emotional_state_history.pop(0)
        self.emotional_state_history.append({
            'score': current_score,
            'label': current_label,
            'timestamp': datetime.now()
        })
        
        if len(self.emotional_state_history) < 2:
            return {
                'current_state': current_label,
                'stability': 'unknown',
                'trajectory': 'unknown',
                'pattern_detected': None
            }
        
        # Calculate stability
        recent_scores = [state['score'] for state in self.emotional_state_history[-3:]]
        stability_score = 1.0 - np.std(recent_scores)  # Lower std = more stable
        
        if stability_score > 0.85:
            stability = 'stable'
        elif stability_score > 0.7:
            stability = 'moderate'
        else:
            stability = 'volatile'
        
        # Calculate trajectory
        if len(self.emotional_state_history) >= 3:
            scores = [state['score'] for state in self.emotional_state_history[-3:]]
            if scores[-1] > scores[0] + 0.1:
                trajectory = 'improving'
            elif scores[-1] < scores[0] - 0.1:
                trajectory = 'declining'
            else:
                trajectory = 'stable'
        else:
            trajectory = 'stable'
        
        # Pattern detection
        pattern = self._detect_specific_patterns()
        
        return {
            'current_state': current_label,
            'stability': stability,
            'trajectory': trajectory,
            'pattern_detected': pattern,
            'stability_score': stability_score
        }
    
    def _generate_insights(self, unified_sentiment: Dict, crisis_assessment: Dict, 
                          emotional_patterns: Dict, modalities: List[str]) -> List[str]:
        """Generate actionable insights from fused analysis"""
        insights = []
        
        # Sentiment-based insights
        sentiment_score = unified_sentiment['score']
        sentiment_label = unified_sentiment['label']
        confidence = unified_sentiment['confidence']
        
        if confidence > 0.8:
            if sentiment_score < -0.5:
                insights.append(f"High confidence detection of {sentiment_label} emotional state across multiple modalities")
            elif sentiment_score > 0.5:
                insights.append(f"Strong positive emotional indicators detected with high confidence")
        
        # Crisis-related insights
        if crisis_assessment['risk_score'] > 0.6:
            insights.append(f"Elevated crisis risk detected: {crisis_assessment['risk_level']} level")
            if crisis_assessment['indicators']:
                insights.append(f"Key crisis indicators: {', '.join(crisis_assessment['indicators'][:3])}")
        
        # Pattern-based insights
        trajectory = emotional_patterns['trajectory']
        stability = emotional_patterns['stability']
        
        if trajectory == 'declining' and stability == 'volatile':
            insights.append("Declining emotional trajectory with high volatility - may benefit from stabilization techniques")
        elif trajectory == 'improving':
            insights.append("Positive emotional trajectory detected - current interventions may be effective")
        
        # Modality-specific insights
        if len(modalities) > 1:
            insights.append(f"Multimodal analysis provides enhanced reliability (using {', '.join(modalities)})")
        
        # Inconsistency warnings
        consistency = unified_sentiment.get('consistency_factor', 1.0)
        if consistency < 0.7:
            insights.append("Emotional expression inconsistency detected across modalities - may indicate emotional complexity")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _calculate_recommendation_priority(self, crisis_assessment: Dict, emotional_patterns: Dict) -> str:
        """Calculate priority level for recommendations"""
        crisis_risk = crisis_assessment['risk_score']
        trajectory = emotional_patterns['trajectory']
        stability = emotional_patterns['stability']
        
        if crisis_risk > 0.7:
            return 'immediate'
        elif crisis_risk > 0.5 or (trajectory == 'declining' and stability == 'volatile'):
            return 'high'
        elif trajectory == 'declining' or stability == 'volatile':
            return 'medium'
        else:
            return 'low'
    
    def _calculate_modality_contributions(self, extracted_metrics: Dict) -> Dict:
        """Calculate how much each modality contributed to final assessment"""
        contributions = {}
        
        for modality, data in extracted_metrics.items():
            weight = self.modality_weights.get(modality, 0.33)
            confidence = data.get('confidence', 0.5)
            contribution = weight * confidence
            contributions[modality] = contribution
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: v/total_contribution for k, v in contributions.items()}
        
        return contributions
    
    # Helper methods
    def _default_analysis(self) -> Dict:
        """Return default analysis when no modalities available"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'sentiment_confidence': 0.0,
            'crisis_risk': 0.0,
            'crisis_indicators': [],
            'confidence_scores': {},
            'overall_confidence': 0.0,
            'emotional_state': 'neutral',
            'insights': ['No modality data available for analysis']
        }
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.6:
            return 'very_positive'
        elif score > 0.2:
            return 'positive'
        elif score > -0.2:
            return 'neutral'
        elif score > -0.6:
            return 'negative'
        else:
            return 'very_negative'
    
    def _calculate_consistency_factor(self, scores: List[float]) -> float:
        """Calculate consistency factor based on score variance"""
        if len(scores) < 2:
            return 1.0
        
        std_dev = np.std(scores)
        # Lower standard deviation = higher consistency
        consistency = max(0.5, 1.0 - std_dev)
        return consistency
    
    def _get_consistency_bonus(self, modality: str, extracted_metrics: Dict) -> float:
        """Get consistency bonus for modality agreement"""
        # Implementation for cross-modality consistency
        return 0.1  # Placeholder
    
    def _get_historical_consistency_bonus(self, modality: str) -> float:
        """Get bonus for consistency with historical patterns"""
        # Implementation for historical consistency
        return 0.05  # Placeholder
    
    def _get_context_appropriateness_bonus(self, modality: str, modality_results: Dict) -> float:
        """Get bonus for contextual appropriateness"""
        # Implementation for context appropriateness
        return 0.05  # Placeholder
    
    def _detect_specific_patterns(self) -> Optional[str]:
        """Detect specific emotional patterns"""
        if len(self.emotional_state_history) < 4:
            return None
        
        recent_labels = [state['label'] for state in self.emotional_state_history[-4:]]
        
        # Pattern detection logic
        if all(label in ['negative', 'very_negative'] for label in recent_labels):
            return 'persistent_negativity'
        elif len(set(recent_labels)) == len(recent_labels):
            return 'emotional_volatility'
        elif recent_labels[0] != recent_labels[-1] and 'positive' in recent_labels[-1]:
            return 'emotional_recovery'
        
        return None
    
    def _update_emotional_history(self, fused_result: Dict):
        """Update emotional state history with fused results"""
        # Update pattern cache and learning algorithms
        pass
    
    def get_fusion_statistics(self) -> Dict:
        """Get statistics about fusion performance"""
        return {
            'total_analyses': len(self.emotional_state_history),
            'modality_weights': self.modality_weights,
            'average_confidence': np.mean([state.get('confidence', 0) 
                                         for state in self.emotional_state_history[-10:]]) if self.emotional_state_history else 0,
            'pattern_cache_size': len(self.pattern_cache)
        }
    
    def update_modality_weights(self, new_weights: Dict[str, float]):
        """Update modality weights based on performance feedback"""
        total_weight = sum(new_weights.values())
        self.modality_weights = {k: v/total_weight for k, v in new_weights.items()}
    
    def reset_history(self):
        """Reset emotional state history"""
        self.emotional_state_history = []
        self.pattern_cache = {}