"""
Visual Analyzer for Multimodal AI Therapy System
Handles facial expression recognition, micro-expressions, eye tracking, and body language analysis
Uses MediaPipe for comprehensive visual emotion detection
"""

import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from PIL import Image
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmotionScore:
    """Data class for emotion detection results"""
    emotion: str
    confidence: float
    timestamp: datetime
    facial_landmarks: Optional[Dict] = None
    micro_expressions: Optional[Dict] = None

@dataclass
class VisualMetrics:
    """Comprehensive visual analysis metrics"""
    primary_emotion: str
    emotion_confidence: float
    stress_level: float
    engagement_score: float
    attention_score: float
    micro_expression_count: int
    eye_contact_ratio: float
    body_language_openness: float
    crisis_indicators: List[str]
    timestamp: datetime

class VisualAnalyzer:
    """
    Comprehensive visual analysis system for therapy sessions
    Processes facial expressions, body language, and behavioral cues
    """
    
    def __init__(self):
        """Initialize MediaPipe components and emotion models"""
        self.setup_mediapipe()
        self.setup_emotion_detection()
        self.setup_crisis_indicators()
        self.initialize_tracking_variables()
        
    def setup_mediapipe(self):
        """Initialize MediaPipe solutions"""
        try:
            # Face detection and landmarks
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.7
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # Pose detection for body language
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # Hands detection for gestures
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # Drawing utilities
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info("✅ MediaPipe components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            raise
    
    def setup_emotion_detection(self):
        """Setup emotion detection using facial landmark analysis"""
        # Define facial landmark indices for emotion detection
        self.emotion_landmarks = {
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # Eyebrow points
            'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],  # Eye contours
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],  # Mouth corners and shape
            'cheeks': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 187],  # Cheek area
            'forehead': [9, 10, 151, 337, 299, 333, 298, 301]  # Forehead region
        }
        
        # Emotion classification thresholds
        self.emotion_thresholds = {
            'happy': {'mouth_curve': 0.02, 'eye_squeeze': 0.01, 'cheek_raise': 0.015},
            'sad': {'mouth_curve': -0.02, 'eyebrow_inner': 0.02, 'eye_droop': 0.01},
            'angry': {'eyebrow_lower': 0.02, 'eye_narrow': 0.015, 'mouth_tight': 0.01},
            'surprised': {'eyebrow_raise': 0.025, 'eye_wide': 0.02, 'mouth_open': 0.03},
            'fearful': {'eyebrow_raise': 0.02, 'eye_wide': 0.025, 'mouth_slight_open': 0.015},
            'disgusted': {'nose_wrinkle': 0.015, 'upper_lip_raise': 0.02, 'eyebrow_lower': 0.01},
            'contempt': {'mouth_corner_up_unilateral': 0.015, 'eye_slight_narrow': 0.01}
        }
        
        logger.info("✅ Emotion detection system configured")
    
    def setup_crisis_indicators(self):
        """Define visual indicators that may suggest crisis situations"""
        self.crisis_indicators = {
            'excessive_distress': {
                'crying_indicators': ['excessive_eye_moisture', 'face_contortion', 'hand_to_face'],
                'agitation_signs': ['rapid_head_movements', 'hand_fidgeting', 'body_rocking'],
                'withdrawal_signs': ['face_covering', 'body_turning_away', 'minimal_eye_contact']
            },
            'self_harm_risk': {
                'concerning_gestures': ['hand_to_wrist', 'scratching_motions', 'head_hitting'],
                'body_language': ['protective_posturing', 'self_soothing_excess', 'tension_indicators']
            },
            'dissociation_signs': {
                'eye_indicators': ['blank_stare', 'unfocused_gaze', 'minimal_blinking'],
                'facial_signs': ['flat_affect', 'minimal_expression_change', 'jaw_tension'],
                'body_signs': ['rigid_posture', 'minimal_movement', 'disconnected_gestures']
            }
        }
        
        logger.info("✅ Crisis indicator system configured")
    
    def initialize_tracking_variables(self):
        """Initialize variables for continuous tracking"""
        self.emotion_history = []
        self.stress_indicators = []
        self.engagement_metrics = []
        self.micro_expression_buffer = []
        self.baseline_measurements = None
        self.frame_count = 0
        self.session_start = datetime.now()
        
        # Calibration variables
        self.calibration_frames = 0
        self.calibration_complete = False
        self.baseline_face_measurements = {}
        
        logger.info("✅ Tracking variables initialized")
    
    def analyze_frame(self, frame: np.ndarray) -> VisualMetrics:
        """
        Analyze a single frame for comprehensive visual metrics
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            VisualMetrics: Comprehensive analysis results
        """
        self.frame_count += 1
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get all MediaPipe analyses
            face_results = self.face_mesh.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            # Analyze different components
            emotion_analysis = self.analyze_facial_emotions(face_results, frame.shape)
            stress_analysis = self.analyze_stress_indicators(face_results, pose_results)
            engagement_analysis = self.analyze_engagement(face_results, pose_results)
            micro_expressions = self.detect_micro_expressions(face_results)
            crisis_indicators = self.detect_crisis_indicators(face_results, pose_results, hand_results)
            
            # Calculate comprehensive metrics
            metrics = VisualMetrics(
                primary_emotion=emotion_analysis['primary_emotion'],
                emotion_confidence=emotion_analysis['confidence'],
                stress_level=stress_analysis['stress_level'],
                engagement_score=engagement_analysis['engagement_score'],
                attention_score=engagement_analysis['attention_score'],
                micro_expression_count=len(micro_expressions),
                eye_contact_ratio=engagement_analysis['eye_contact_ratio'],
                body_language_openness=self.analyze_body_openness(pose_results),
                crisis_indicators=crisis_indicators,
                timestamp=datetime.now()
            )
            
            # Update history
            self.emotion_history.append(emotion_analysis)
            self.stress_indicators.append(stress_analysis)
            self.engagement_metrics.append(engagement_analysis)
            
            # Maintain rolling history (keep last 100 frames)
            if len(self.emotion_history) > 100:
                self.emotion_history.pop(0)
                self.stress_indicators.pop(0)
                self.engagement_metrics.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Frame analysis failed: {e}")
            return self.get_default_metrics()
    
    def analyze_facial_emotions(self, face_results, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Analyze facial expressions for emotion detection"""
        
        if not face_results.multi_face_landmarks:
            return {
                'primary_emotion': 'neutral',
                'confidence': 0.0,
                'emotion_scores': {},
                'facial_measurements': {}
            }
        
        landmarks = face_results.multi_face_landmarks[0]
        h, w = frame_shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for lm in landmarks.landmark:
            landmark_points.append([int(lm.x * w), int(lm.y * h), lm.z])
        
        landmark_points = np.array(landmark_points)
        
        # Calculate facial measurements for emotion detection
        measurements = self.calculate_facial_measurements(landmark_points)
        
        # Establish baseline if not done
        if not self.calibration_complete:
            self.update_baseline_measurements(measurements)
        
        # Detect emotions based on facial measurements
        emotion_scores = self.classify_emotions(measurements)
        
        # Get primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'emotion_scores': emotion_scores,
            'facial_measurements': measurements,
            'landmark_count': len(landmark_points)
        }
    
    def calculate_facial_measurements(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate key facial measurements for emotion detection"""
        
        measurements = {}
        
        try:
            # Mouth measurements
            mouth_left = landmarks[61][:2]
            mouth_right = landmarks[291][:2]
            mouth_top = landmarks[13][:2]
            mouth_bottom = landmarks[14][:2]
            
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Mouth curvature (happiness/sadness indicator)
            mouth_curve = (landmarks[308][1] + landmarks[78][1]) / 2 - landmarks[13][1]
            measurements['mouth_curve'] = mouth_curve / mouth_width if mouth_width > 0 else 0
            measurements['mouth_aspect_ratio'] = mouth_aspect_ratio
            
            # Eye measurements
            # Left eye
            left_eye_points = landmarks[[33, 7, 163, 144, 145, 153]][:, :2]
            left_eye_center = np.mean(left_eye_points, axis=0)
            left_eye_width = np.linalg.norm(landmarks[33][:2] - landmarks[133][:2])
            left_eye_height = np.linalg.norm(landmarks[159][:2] - landmarks[145][:2])
            left_eye_aspect_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            
            # Right eye
            right_eye_points = landmarks[[362, 382, 381, 380, 374, 373]][:, :2]
            right_eye_center = np.mean(right_eye_points, axis=0)
            right_eye_width = np.linalg.norm(landmarks[362][:2] - landmarks[263][:2])
            right_eye_height = np.linalg.norm(landmarks[386][:2] - landmarks[374][:2])
            right_eye_aspect_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            
            measurements['left_eye_aspect_ratio'] = left_eye_aspect_ratio
            measurements['right_eye_aspect_ratio'] = right_eye_aspect_ratio
            measurements['average_eye_aspect_ratio'] = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2
            
            # Eyebrow measurements
            left_eyebrow_height = landmarks[70][1] - left_eye_center[1]
            right_eyebrow_height = landmarks[300][1] - right_eye_center[1]
            measurements['eyebrow_raise'] = (left_eyebrow_height + right_eyebrow_height) / 2
            
            # Face dimensions for normalization
            face_width = np.linalg.norm(landmarks[454][:2] - landmarks[234][:2])
            face_height = np.linalg.norm(landmarks[10][:2] - landmarks[152][:2])
            measurements['face_width'] = face_width
            measurements['face_height'] = face_height
            
            # Normalize measurements by face size
            for key in ['mouth_curve', 'eyebrow_raise']:
                if key in measurements and face_height > 0:
                    measurements[f'{key}_normalized'] = measurements[key] / face_height
            
        except Exception as e:
            logger.warning(f"⚠️  Facial measurement calculation failed: {e}")
            measurements = self.get_default_measurements()
        
        return measurements
    
    def classify_emotions(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """Classify emotions based on facial measurements"""
        
        emotion_scores = {
            'neutral': 0.5,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'fearful': 0.0,
            'disgusted': 0.0
        }
        
        try:
            # Happiness detection
            if measurements.get('mouth_curve_normalized', 0) > 0.01:
                emotion_scores['happy'] += measurements['mouth_curve_normalized'] * 10
                
            if measurements.get('average_eye_aspect_ratio', 0) < 0.25:  # Squinting from smiling
                emotion_scores['happy'] += 0.2
            
            # Sadness detection
            if measurements.get('mouth_curve_normalized', 0) < -0.01:
                emotion_scores['sad'] += abs(measurements['mouth_curve_normalized']) * 10
                
            if measurements.get('eyebrow_raise_normalized', 0) > 0.02:  # Inner brow raise
                emotion_scores['sad'] += 0.3
            
            # Surprise detection
            if measurements.get('eyebrow_raise_normalized', 0) > 0.03:
                emotion_scores['surprised'] += measurements['eyebrow_raise_normalized'] * 8
                
            if measurements.get('average_eye_aspect_ratio', 0) > 0.35:  # Wide eyes
                emotion_scores['surprised'] += 0.4
                
            if measurements.get('mouth_aspect_ratio', 0) > 0.7:  # Open mouth
                emotion_scores['surprised'] += 0.3
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
        except Exception as e:
            logger.warning(f"⚠️  Emotion classification failed: {e}")
        
        return emotion_scores
    
    def analyze_stress_indicators(self, face_results, pose_results) -> Dict[str, Any]:
        """Analyze visual indicators of stress and anxiety"""
        
        stress_level = 0.0
        stress_factors = []
        
        try:
            # Facial tension indicators
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0]
                
                # Jaw tension (distance between jaw points)
                jaw_tension = self.calculate_jaw_tension(landmarks)
                if jaw_tension > 0.8:  # Threshold for tension
                    stress_level += 0.3
                    stress_factors.append("jaw_tension")
                
                # Eye strain (blink rate and eye openness)
                eye_strain = self.calculate_eye_strain(landmarks)
                if eye_strain > 0.7:
                    stress_level += 0.2
                    stress_factors.append("eye_strain")
                
                # Forehead tension
                forehead_tension = self.calculate_forehead_tension(landmarks)
                if forehead_tension > 0.6:
                    stress_level += 0.2
                    stress_factors.append("forehead_tension")
            
            # Body posture stress indicators
            if pose_results.pose_landmarks:
                # Shoulder tension (shoulder height asymmetry)
                shoulder_tension = self.calculate_shoulder_tension(pose_results.pose_landmarks)
                if shoulder_tension > 0.7:
                    stress_level += 0.2
                    stress_factors.append("shoulder_tension")
                
                # Head position (forward head posture indicates stress)
                head_posture = self.calculate_head_posture(pose_results.pose_landmarks)
                if head_posture > 0.6:
                    stress_level += 0.1
                    stress_factors.append("forward_head_posture")
            
            # Clamp stress level between 0 and 1
            stress_level = min(1.0, max(0.0, stress_level))
            
        except Exception as e:
            logger.warning(f"⚠️  Stress analysis failed: {e}")
        
        return {
            'stress_level': stress_level,
            'stress_factors': stress_factors,
            'timestamp': datetime.now()
        }
    
    def analyze_engagement(self, face_results, pose_results) -> Dict[str, Any]:
        """Analyze user engagement and attention levels"""
        
        engagement_score = 0.5  # Baseline
        attention_score = 0.5
        eye_contact_ratio = 0.0
        
        try:
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0]
                
                # Eye contact estimation (gaze direction)
                gaze_direction = self.estimate_gaze_direction(landmarks)
                if abs(gaze_direction['horizontal']) < 15 and abs(gaze_direction['vertical']) < 10:
                    eye_contact_ratio = 1.0
                    attention_score += 0.3
                
                # Face orientation (looking at camera)
                face_orientation = self.calculate_face_orientation(landmarks)
                if abs(face_orientation['yaw']) < 20:  # Face towards camera
                    engagement_score += 0.2
                
                # Facial expression variability (engaged people show more expression)
                if len(self.emotion_history) > 10:
                    expression_variance = self.calculate_expression_variance()
                    if expression_variance > 0.1:  # Some variability indicates engagement
                        engagement_score += 0.2
            
            # Body orientation
            if pose_results.pose_landmarks:
                body_orientation = self.calculate_body_orientation(pose_results.pose_landmarks)
                if abs(body_orientation) < 30:  # Body facing forward
                    engagement_score += 0.1
            
            # Clamp scores
            engagement_score = min(1.0, max(0.0, engagement_score))
            attention_score = min(1.0, max(0.0, attention_score))
            
        except Exception as e:
            logger.warning(f"⚠️  Engagement analysis failed: {e}")
        
        return {
            'engagement_score': engagement_score,
            'attention_score': attention_score,
            'eye_contact_ratio': eye_contact_ratio,
            'timestamp': datetime.now()
        }
    
    def detect_micro_expressions(self, face_results) -> List[Dict[str, Any]]:
        """Detect brief micro-expressions that may indicate suppressed emotions"""
        
        micro_expressions = []
        
        try:
            if face_results.multi_face_landmarks and len(self.emotion_history) > 5:
                current_emotion = self.emotion_history[-1]
                
                # Check for rapid emotion changes (micro-expressions)
                for i in range(max(0, len(self.emotion_history) - 5), len(self.emotion_history) - 1):
                    prev_emotion = self.emotion_history[i]
                    
                    # Calculate emotion change magnitude
                    emotion_change = 0
                    for emotion in current_emotion['emotion_scores']:
                        change = abs(current_emotion['emotion_scores'][emotion] - 
                                   prev_emotion['emotion_scores'][emotion])
                        emotion_change += change
                    
                    # If significant rapid change detected
                    if emotion_change > 0.3:  # Threshold for micro-expression
                        micro_expressions.append({
                            'type': 'rapid_emotion_change',
                            'magnitude': emotion_change,
                            'from_emotion': prev_emotion['primary_emotion'],
                            'to_emotion': current_emotion['primary_emotion'],
                            'frame_gap': len(self.emotion_history) - 1 - i,
                            'timestamp': datetime.now()
                        })
                
        except Exception as e:
            logger.warning(f"⚠️  Micro-expression detection failed: {e}")
        
        return micro_expressions
    
    def detect_crisis_indicators(self, face_results, pose_results, hand_results) -> List[str]:
        """Detect visual indicators that may suggest crisis situations"""
        
        crisis_indicators = []
        
        try:
            # Facial distress indicators
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0]
                
                # Check for extreme expressions
                if len(self.emotion_history) > 0:
                    current_emotion = self.emotion_history[-1]
                    
                    # Extreme sadness
                    if current_emotion['emotion_scores'].get('sad', 0) > 0.8:
                        crisis_indicators.append("extreme_sadness")
                    
                    # Signs of crying or extreme distress
                    if self.detect_crying_indicators(landmarks):
                        crisis_indicators.append("crying_distress")
                
                # Flat affect (potential dissociation)
                if self.detect_flat_affect():
                    crisis_indicators.append("flat_affect")
            
            # Body language crisis indicators
            if pose_results.pose_landmarks:
                # Self-protective postures
                if self.detect_protective_postures(pose_results.pose_landmarks):
                    crisis_indicators.append("protective_posturing")
                
                # Signs of agitation
                if self.detect_agitation_signs(pose_results.pose_landmarks):
                    crisis_indicators.append("agitation")
            
            # Hand gesture analysis
            if hand_results.multi_hand_landmarks:
                concerning_gestures = self.analyze_concerning_gestures(hand_results.multi_hand_landmarks)
                crisis_indicators.extend(concerning_gestures)
            
        except Exception as e:
            logger.warning(f"⚠️  Crisis indicator detection failed: {e}")
        
        return crisis_indicators
    
    def analyze_body_openness(self, pose_results) -> float:
        """Analyze body language for openness/defensiveness"""
        
        openness_score = 0.5  # Baseline neutral
        
        try:
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Arm position analysis
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                
                # Calculate arm openness (arms spread = more open)
                shoulder_width = abs(right_shoulder.x - left_shoulder.x)
                arm_spread = (abs(left_elbow.x - left_shoulder.x) + 
                            abs(right_elbow.x - right_shoulder.x)) / 2
                
                if shoulder_width > 0:
                    arm_openness_ratio = arm_spread / shoulder_width
                    openness_score += min(0.3, arm_openness_ratio * 0.5)
                
                # Torso orientation (facing forward = more open)
                torso_angle = math.atan2(right_shoulder.y - left_shoulder.y,
                                       right_shoulder.x - left_shoulder.x)
                torso_openness = 1.0 - abs(torso_angle) / (math.pi / 2)
                openness_score += torso_openness * 0.2
                
        except Exception as e:
            logger.warning(f"⚠️  Body openness analysis failed: {e}")
        
        return min(1.0, max(0.0, openness_score))
    
    # Helper methods for specific calculations
    
    def calculate_jaw_tension(self, landmarks) -> float:
        """Calculate jaw tension based on landmark positions"""
        try:
            # Approximate jaw tension by measuring jaw width
            jaw_left = landmarks.landmark[172]
            jaw_right = landmarks.landmark[397]
            jaw_width = abs(jaw_right.x - jaw_left.x)
            
            # Normalize and return tension score (smaller width = more tension)
            return max(0, (0.1 - jaw_width) * 10)  # Adjust threshold as needed
        except:
            return 0.0
    
    def calculate_eye_strain(self, landmarks) -> float:
        """Calculate eye strain indicators"""
        try:
            # Eye aspect ratio for both eyes
            left_ear = self.calculate_eye_aspect_ratio(landmarks, 'left')
            right_ear = self.calculate_eye_aspect_ratio(landmarks, 'right')
            avg_ear = (left_ear + right_ear) / 2
            
            # Lower EAR indicates more strain/squinting
            strain_score = max(0, (0.25 - avg_ear) * 4)  # Normalize
            return min(1.0, strain_score)
        except:
            return 0.0
    
    def calculate_eye_aspect_ratio(self, landmarks, eye: str) -> float:
        """Calculate eye aspect ratio for strain detection"""
        try:
            if eye == 'left':
                # Left eye landmarks
                p1, p2 = landmarks.landmark[159], landmarks.landmark[145]  # Vertical
                p3, p4 = landmarks.landmark[33], landmarks.landmark[133]   # Horizontal
            else:
                # Right eye landmarks  
                p1, p2 = landmarks.landmark[386], landmarks.landmark[374]  # Vertical
                p3, p4 = landmarks.landmark[362], landmarks.landmark[263]  # Horizontal
            
            # Calculate distances
            vertical_dist = abs(p1.y - p2.y)
            horizontal_dist = abs(p4.x - p3.x)
            
            # Return aspect ratio
            return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
        except:
            return 0.25  # Default normal EAR
    
    def get_default_metrics(self) -> VisualMetrics:
        """Return default metrics when analysis fails"""
        return VisualMetrics(
            primary_emotion='neutral',
            emotion_confidence=0.0,
            stress_level=0.0,
            engagement_score=0.5,
            attention_score=0.5,
            micro_expression_count=0,
            eye_contact_ratio=0.0,
            body_language_openness=0.5,
            crisis_indicators=[],
            timestamp=datetime.now()
        )
    
    def get_default_measurements(self) -> Dict[str, float]:
        """Return default facial measurements"""
        return {
            'mouth_curve': 0.0,
            'mouth_aspect_ratio': 0.5,
            'left_eye_aspect_ratio': 0.25,
            'right_eye_aspect_ratio': 0.25,
            'average_eye_aspect_ratio': 0.25,
            'eyebrow_raise': 0.0,
            'face_width': 100.0,
            'face_height': 120.0
        }
    
    def analyze(self, video_data: np.ndarray) -> Dict[str, Any]:
        """
        Main analysis method for video data
        
        Args:
            video_data: Video frame or sequence as numpy array
            
        Returns:
            Dict containing comprehensive visual analysis results
        """
        try:
            # Handle single frame vs video sequence
            if len(video_data.shape) == 3:
                # Single frame
                metrics = self.analyze_frame(video_data)
            else:
                # Video sequence - analyze last frame for real-time
                metrics = self.analyze_frame(video_data[-1])
            
            # Convert to dictionary for consistency with other analyzers
            analysis_result = {
                'primary_emotion': metrics.primary_emotion,
                'emotion_confidence': metrics.emotion_confidence,
                'stress_level': metrics.stress_level,
                'engagement_score': metrics.engagement_score,
                'attention_score': metrics.attention_score,
                'micro_expression_count': metrics.micro_expression_count,
                'eye_contact_ratio': metrics.eye_contact_ratio,
                'body_language_openness': metrics.body_language_openness,
                'crisis_indicators': metrics.crisis_indicators,
                'analysis_type': 'visual',
                'timestamp': metrics.timestamp,
                'frame_count': self.frame_count,
                'session_duration': (datetime.now() - self.session_start).total_seconds()
            }
            
            # Add emotion history summary
            if len(self.emotion_history) > 0:
                recent_emotions = [e['primary_emotion'] for e in self.emotion_history[-10:]]
                emotion_stability = len(set(recent_emotions)) / len(recent_emotions)
                analysis_result['emotion_stability'] = 1.0 - emotion_stability
            
            # Add trend analysis
            if len(self.stress_indicators) > 5:
                recent_stress = [s['stress_level'] for s in self.stress_indicators[-5:]]
                stress_trend = np.polyfit(range(len(recent_stress)), recent_stress, 1)[0]
                analysis_result['stress_trend'] = stress_trend  # Positive = increasing stress
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Visual analysis failed: {e}")
            return {
                'primary_emotion': 'neutral',
                'emotion_confidence': 0.0,
                'stress_level': 0.0,
                'engagement_score': 0.5,
                'attention_score': 0.5,
                'crisis_indicators': [],
                'analysis_type': 'visual',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def update_baseline_measurements(self, measurements: Dict[str, float]):
        """Update baseline facial measurements during calibration"""
        if self.calibration_frames < 30:  # Calibrate for 30 frames
            if not self.baseline_face_measurements:
                self.baseline_face_measurements = measurements.copy()
            else:
                # Running average
                for key, value in measurements.items():
                    if key in self.baseline_face_measurements:
                        self.baseline_face_measurements[key] = (
                            (self.baseline_face_measurements[key] * self.calibration_frames + value) /
                            (self.calibration_frames + 1)
                        )
            
            self.calibration_frames += 1
            
        elif self.calibration_frames >= 30:
            self.calibration_complete = True
            logger.info("✅ Facial baseline calibration complete")
    
    def estimate_gaze_direction(self, landmarks) -> Dict[str, float]:
        """Estimate gaze direction from facial landmarks"""
        try:
            # Get eye centers and corners
            left_eye_center = np.array([landmarks.landmark[468].x, landmarks.landmark[468].y])
            right_eye_center = np.array([landmarks.landmark[473].x, landmarks.landmark[473].y])
            
            # Estimate gaze based on eye position relative to eye corners
            left_corner = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y])
            right_corner = np.array([landmarks.landmark[263].x, landmarks.landmark[263].y])
            
            # Calculate horizontal gaze angle
            eye_line_vector = right_corner - left_corner
            gaze_vector = (left_eye_center + right_eye_center) / 2 - (left_corner + right_corner) / 2
            
            horizontal_angle = np.arctan2(gaze_vector[0], eye_line_vector[0]) * 180 / np.pi
            vertical_angle = np.arctan2(gaze_vector[1], np.linalg.norm(eye_line_vector)) * 180 / np.pi
            
            return {
                'horizontal': horizontal_angle,
                'vertical': vertical_angle
            }
        except:
            return {'horizontal': 0.0, 'vertical': 0.0}
    
    def calculate_face_orientation(self, landmarks) -> Dict[str, float]:
        """Calculate face orientation (yaw, pitch, roll)"""
        try:
            # Use nose tip, chin, and forehead points for orientation
            nose_tip = np.array([landmarks.landmark[1].x, landmarks.landmark[1].y, landmarks.landmark[1].z])
            chin = np.array([landmarks.landmark[18].x, landmarks.landmark[18].y, landmarks.landmark[18].z])
            forehead = np.array([landmarks.landmark[10].x, landmarks.landmark[10].y, landmarks.landmark[10].z])
            
            # Calculate yaw (left-right rotation)
            left_cheek = np.array([landmarks.landmark[234].x, landmarks.landmark[234].y])
            right_cheek = np.array([landmarks.landmark[454].x, landmarks.landmark[454].y])
            
            face_width_vector = right_cheek - left_cheek
            yaw = np.arctan2(face_width_vector[1], face_width_vector[0]) * 180 / np.pi
            
            # Calculate pitch (up-down rotation)
            face_height_vector = chin[:2] - forehead[:2]
            pitch = np.arctan2(face_height_vector[0], face_height_vector[1]) * 180 / np.pi
            
            return {
                'yaw': yaw,
                'pitch': pitch,
                'roll': 0.0  # Simplified - roll calculation requires more complex 3D analysis
            }
        except:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    def calculate_expression_variance(self) -> float:
        """Calculate variance in facial expressions over recent history"""
        try:
            if len(self.emotion_history) < 5:
                return 0.0
            
            # Get emotion scores for recent frames
            recent_emotions = self.emotion_history[-10:]
            emotion_vectors = []
            
            for emotion_data in recent_emotions:
                vector = list(emotion_data['emotion_scores'].values())
                emotion_vectors.append(vector)
            
            emotion_vectors = np.array(emotion_vectors)
            
            # Calculate variance across time
            variance = np.var(emotion_vectors, axis=0).mean()
            return float(variance)
            
        except:
            return 0.0
    
    def calculate_body_orientation(self, pose_landmarks) -> float:
        """Calculate body orientation angle"""
        try:
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate angle of shoulder line
            shoulder_angle = math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ) * 180 / math.pi
            
            return shoulder_angle
        except:
            return 0.0
    
    def calculate_forehead_tension(self, landmarks) -> float:
        """Calculate forehead tension from landmark positions"""
        try:
            # Get forehead points
            forehead_points = [landmarks.landmark[9], landmarks.landmark[10], 
                             landmarks.landmark[151], landmarks.landmark[337]]
            
            # Calculate forehead area (tension reduces area)
            points = np.array([[p.x, p.y] for p in forehead_points])
            
            # Simple area calculation using cross product
            area = 0.5 * abs(
                (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) -
                (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
            )
            
            # Normalized tension score (smaller area = more tension)
            baseline_area = 0.01  # Approximate normal forehead area
            tension = max(0, (baseline_area - area) * 50)
            
            return min(1.0, tension)
        except:
            return 0.0
    
    def calculate_shoulder_tension(self, pose_landmarks) -> float:
        """Calculate shoulder tension from pose landmarks"""
        try:
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            neck = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]  # Approximate neck position
            
            # Calculate shoulder height asymmetry
            shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Calculate shoulder elevation (closer to neck = more tense)
            left_elevation = abs(left_shoulder.y - neck.y)
            right_elevation = abs(right_shoulder.y - neck.y)
            avg_elevation = (left_elevation + right_elevation) / 2
            
            # Combine metrics for tension score
            asymmetry_score = shoulder_height_diff * 10  # Scale factor
            elevation_score = max(0, (0.1 - avg_elevation) * 10)  # Elevated shoulders
            
            tension = (asymmetry_score + elevation_score) / 2
            return min(1.0, tension)
        except:
            return 0.0
    
    def calculate_head_posture(self, pose_landmarks) -> float:
        """Calculate forward head posture indicator"""
        try:
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate shoulder midpoint
            shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
            
            # Forward head posture = head significantly forward of shoulders
            head_forward_distance = abs(nose.x - shoulder_midpoint_x)
            
            # Normalize (typical forward head posture threshold)
            posture_score = head_forward_distance * 5  # Scale factor
            
            return min(1.0, posture_score)
        except:
            return 0.0
    
    def detect_crying_indicators(self, landmarks) -> bool:
        """Detect visual indicators of crying"""
        try:
            # Check for eye puffiness/closure patterns
            left_ear = self.calculate_eye_aspect_ratio(landmarks, 'left')
            right_ear = self.calculate_eye_aspect_ratio(landmarks, 'right')
            avg_ear = (left_ear + right_ear) / 2
            
            # Crying often involves partially closed eyes
            if avg_ear < 0.15:  # Very low EAR
                return True
            
            # Check for mouth distortion (crying mouth shape)
            mouth_landmarks = [landmarks.landmark[61], landmarks.landmark[291], 
                             landmarks.landmark[13], landmarks.landmark[14]]
            
            # Calculate mouth distortion
            mouth_width = abs(mouth_landmarks[1].x - mouth_landmarks[0].x)
            mouth_height = abs(mouth_landmarks[3].y - mouth_landmarks[2].y)
            
            if mouth_height / mouth_width > 0.8:  # Distorted mouth shape
                return True
            
            return False
        except:
            return False
    
    def detect_flat_affect(self) -> bool:
        """Detect flat affect (minimal emotional expression)"""
        if len(self.emotion_history) < 10:
            return False
        
        try:
            # Check recent emotion variance
            recent_emotions = self.emotion_history[-10:]
            all_neutral = True
            
            for emotion_data in recent_emotions:
                if emotion_data['primary_emotion'] != 'neutral' or emotion_data['confidence'] > 0.3:
                    all_neutral = False
                    break
            
            # Also check expression variance
            expression_variance = self.calculate_expression_variance()
            
            return all_neutral and expression_variance < 0.05
        except:
            return False
    
    def detect_protective_postures(self, pose_landmarks) -> bool:
        """Detect self-protective body postures"""
        try:
            left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            # Check if hands are near face (protective gesture)
            left_distance_to_face = math.sqrt((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)
            right_distance_to_face = math.sqrt((right_wrist.x - nose.x)**2 + (right_wrist.y - nose.y)**2)
            
            # If either hand is very close to face
            if left_distance_to_face < 0.15 or right_distance_to_face < 0.15:
                return True
            
            # Check for crossed arms (defensive posture)
            left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            
            # Crossed arms indicator
            if left_elbow.x > nose.x and right_elbow.x < nose.x:
                return True
            
            return False
        except:
            return False
    
    def detect_agitation_signs(self, pose_landmarks) -> bool:
        """Detect signs of agitation in body movement"""
        # This would require tracking movement over time
        # For now, check for extreme poses that might indicate agitation
        try:
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Check for extreme shoulder positions
            shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
            
            if shoulder_height_diff > 0.1:  # Significant asymmetry
                return True
            
            return False
        except:
            return False
    
    def analyze_concerning_gestures(self, hand_landmarks_list) -> List[str]:
        """Analyze hand gestures for concerning patterns"""
        concerning_gestures = []
        
        try:
            for hand_landmarks in hand_landmarks_list:
                # Get wrist and fingertip positions
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # Check for repetitive touching gestures (self-soothing when excessive)
                finger_to_wrist_distance = math.sqrt(
                    (index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2
                )
                
                if finger_to_wrist_distance < 0.05:  # Very close - possible self-touching
                    concerning_gestures.append("excessive_self_touch")
        
        except Exception as e:
            logger.warning(f"⚠️  Hand gesture analysis failed: {e}")
        
        return concerning_gestures
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        try:
            if not self.emotion_history:
                return {'error': 'No session data available'}
            
            # Emotion distribution
            all_emotions = [e['primary_emotion'] for e in self.emotion_history]
            emotion_counts = {emotion: all_emotions.count(emotion) for emotion in set(all_emotions)}
            
            # Stress analysis
            all_stress = [s['stress_level'] for s in self.stress_indicators if 'stress_level' in s]
            avg_stress = np.mean(all_stress) if all_stress else 0
            
            # Engagement analysis
            all_engagement = [e['engagement_score'] for e in self.engagement_metrics if 'engagement_score' in e]
            avg_engagement = np.mean(all_engagement) if all_engagement else 0
            
            # Crisis indicators
            all_crisis_indicators = []
            for result in self.emotion_history:
                if hasattr(result, 'crisis_indicators'):
                    all_crisis_indicators.extend(result.crisis_indicators)
            
            crisis_counts = {indicator: all_crisis_indicators.count(indicator) 
                           for indicator in set(all_crisis_indicators)}
            
            return {
                'session_duration': (datetime.now() - self.session_start).total_seconds(),
                'total_frames_analyzed': self.frame_count,
                'emotion_distribution': emotion_counts,
                'average_stress_level': avg_stress,
                'average_engagement': avg_engagement,
                'crisis_indicators_detected': crisis_counts,
                'micro_expressions_detected': len(self.micro_expression_buffer),
                'calibration_completed': self.calibration_complete,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Session summary generation failed: {e}")
            return {'error': str(e)}
    
    def save_session_data(self, filepath: Path):
        """Save session analysis data to file"""
        try:
            session_data = {
                'session_summary': self.get_session_summary(),
                'emotion_history': [
                    {
                        **emotion_data,
                        'timestamp': emotion_data.get('timestamp', datetime.now()).isoformat()
                    }
                    for emotion_data in self.emotion_history
                ],
                'stress_indicators': [
                    {
                        **stress_data,
                        'timestamp': stress_data.get('timestamp', datetime.now()).isoformat()
                    }
                    for stress_data in self.stress_indicators
                ],
                'engagement_metrics': [
                    {
                        **engagement_data,
                        'timestamp': engagement_data.get('timestamp', datetime.now()).isoformat()
                    }
                    for engagement_data in self.engagement_metrics
                ],
                'micro_expressions': [
                    {
                        **micro_expr,
                        'timestamp': micro_expr.get('timestamp', datetime.now()).isoformat()
                    }
                    for micro_expr in self.micro_expression_buffer
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            logger.info(f"✅ Visual analysis session data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save session data: {e}")
    
    def reset_session(self):
        """Reset session data for new session"""
        self.emotion_history = []
        self.stress_indicators = []
        self.engagement_metrics = []
        self.micro_expression_buffer = []
        self.baseline_measurements = None
        self.frame_count = 0
        self.session_start = datetime.now()
        self.calibration_frames = 0
        self.calibration_complete = False
        self.baseline_face_measurements = {}
        
        logger.info("✅ Visual analyzer session reset")