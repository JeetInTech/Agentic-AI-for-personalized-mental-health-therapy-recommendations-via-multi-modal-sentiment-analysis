import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import base64
import io
from PIL import Image

# Transformer models for visual analysis
try:
    from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class VisualAnalyzer:
    """
    Enhanced visual analyzer for facial emotion recognition, body language analysis,
    and engagement detection using MediaPipe and transformer models
    """
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Emotion recognition models
        self.emotion_models = {}
        self.load_emotion_models()
        
        # Facial landmark indices for specific features
        self.landmark_indices = self.initialize_landmark_indices()
        
        # Emotion mapping
        self.emotion_mappings = {
            'LABEL_0': 'angry',
            'LABEL_1': 'disgust', 
            'LABEL_2': 'fear',
            'LABEL_3': 'happy',
            'LABEL_4': 'sad',
            'LABEL_5': 'surprise',
            'LABEL_6': 'neutral',
            'angry': 'angry',
            'disgust': 'disgusted',
            'fear': 'afraid',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprised',
            'neutral': 'neutral'
        }
        
        # Frame analysis parameters
        self.frame_buffer = []
        self.max_buffer_size = 30  # Store last 30 frames for temporal analysis
        
    def load_emotion_models(self):
        """Load facial emotion recognition models"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Try to load a facial emotion recognition model
                try:
                    self.emotion_models['transformer'] = pipeline(
                        "image-classification",
                        model="trpakov/vit-face-expression"
                    )
                    logging.info("Transformer emotion model loaded")
                except Exception as e:
                    logging.warning(f"Could not load transformer emotion model: {e}")
                
                # Alternative model
                try:
                    self.emotion_models['fer_model'] = pipeline(
                        "image-classification",
                        model="dima806/facial_emotions_image_detection"
                    )
                    logging.info("Alternative emotion model loaded")
                except Exception as e:
                    logging.warning(f"Could not load alternative emotion model: {e}")
        
        except Exception as e:
            logging.warning(f"Transformer models not available: {e}")
        
        # If no transformer models available, we'll use MediaPipe + rule-based analysis
        if not self.emotion_models:
            logging.info("Using MediaPipe-based emotion analysis")
    
    def initialize_landmark_indices(self) -> Dict[str, List[int]]:
        """Initialize facial landmark indices for different facial features"""
        return {
            # Eye landmarks
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            
            # Eyebrow landmarks
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            
            # Mouth landmarks
            'mouth_outer': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415],
            
            # Face contour
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162],
            
            # Nose
            'nose': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 440, 344, 278],
        }
    
    def analyze_video(self, video_input: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Comprehensive video analysis for facial emotions and body language
        
        Args:
            video_input: Path to video file, video bytes, or numpy array
            
        Returns:
            Dictionary containing visual analysis results
        """
        
        try:
            # Process video input
            frames = self.extract_frames(video_input)
            
            if not frames:
                return self.empty_analysis()
            
            # Analyze each frame
            frame_analyses = []
            for i, frame in enumerate(frames):
                frame_analysis = self.analyze_frame(frame, frame_number=i)
                if frame_analysis:
                    frame_analyses.append(frame_analysis)
            
            if not frame_analyses:
                return self.empty_analysis()
            
            # Aggregate results across frames
            aggregated_results = self.aggregate_frame_analyses(frame_analyses)
            
            # Temporal analysis
            temporal_analysis = self.analyze_temporal_patterns(frame_analyses)
            
            # Overall assessment
            overall_assessment = self.compute_overall_assessment(
                aggregated_results, temporal_analysis, len(frames)
            )
            
            results = {
                'input_type': type(video_input).__name__,
                'total_frames': len(frames),
                'analyzed_frames': len(frame_analyses),
                'timestamp': datetime.now().isoformat(),
                **aggregated_results,
                **temporal_analysis,
                **overall_assessment
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in video analysis: {e}")
            return self.fallback_analysis()
    
    def analyze_image(self, image_input: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a single image for facial emotions and body language
        
        Args:
            image_input: Path to image file, image bytes, or numpy array
            
        Returns:
            Dictionary containing visual analysis results
        """
        
        try:
            # Load image
            frame = self.load_image(image_input)
            
            if frame is None:
                return self.empty_analysis()
            
            # Analyze the frame
            frame_analysis = self.analyze_frame(frame)
            
            if not frame_analysis:
                return self.empty_analysis()
            
            # Convert frame analysis to image analysis format
            results = {
                'input_type': type(image_input).__name__,
                'timestamp': datetime.now().isoformat(),
                **frame_analysis,
                'analysis_type': 'single_image'
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in image analysis: {e}")
            return self.fallback_analysis()
    
    def extract_frames(self, video_input: Union[str, bytes, np.ndarray]) -> List[np.ndarray]:
        """Extract frames from video input"""
        
        frames = []
        
        try:
            if isinstance(video_input, str):
                # Video file path
                cap = cv2.VideoCapture(video_input)
            elif isinstance(video_input, bytes):
                # Video bytes - save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    tmp_file.write(video_input)
                    cap = cv2.VideoCapture(tmp_file.name)
            else:
                # Assume it's already a numpy array (single frame)
                return [video_input]
            
            # Extract frames (sample every few frames to avoid too many)
            frame_count = 0
            sample_rate = 5  # Sample every 5th frame
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    
                    # Limit total frames to prevent memory issues
                    if len(frames) >= 20:
                        break
                
                frame_count += 1
            
            cap.release()
            
            # Clean up temporary file if created
            if isinstance(video_input, bytes):
                import os
                os.unlink(tmp_file.name)
            
        except Exception as e:
            logging.error(f"Error extracting frames: {e}")
        
        return frames
    
    def load_image(self, image_input: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from various input types"""
        
        try:
            if isinstance(image_input, str):
                # Image file path
                image = cv2.imread(image_input)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, bytes):
                # Image bytes
                nparr = np.frombuffer(image_input, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, np.ndarray):
                # Already a numpy array
                return image_input
            
            else:
                # Try PIL Image
                if hasattr(image_input, 'convert'):
                    pil_image = image_input.convert('RGB')
                    return np.array(pil_image)
            
        except Exception as e:
            logging.error(f"Error loading image: {e}")
        
        return None
    
    def analyze_frame(self, frame: np.ndarray, frame_number: int = 0) -> Dict[str, Any]:
        """Analyze a single frame for facial and body features"""
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
            
            analysis_result = {
                'frame_number': frame_number,
                'frame_shape': frame.shape,
                'timestamp': datetime.now().isoformat()
            }
            
            # Face detection and analysis
            face_analysis = self.analyze_face(frame_rgb)
            analysis_result.update(face_analysis)
            
            # Hand analysis
            hand_analysis = self.analyze_hands(frame_rgb)
            analysis_result.update(hand_analysis)
            
            # Pose analysis
            pose_analysis = self.analyze_pose(frame_rgb)
            analysis_result.update(pose_analysis)
            
            # Overall engagement estimation
            engagement = self.estimate_engagement(analysis_result)
            analysis_result.update(engagement)
            
            return analysis_result
            
        except Exception as e:
            logging.error(f"Error analyzing frame {frame_number}: {e}")
            return {}
    
    def analyze_face(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze facial features and emotions"""
        
        face_results = {
            'face_detected': False,
            'face_confidence': 0.0,
            'facial_emotion': 'neutral',
            'emotion_confidence': 0.0,
            'facial_landmarks': {},
            'micro_expressions': []
        }
        
        try:
            # Face detection
            detection_results = self.face_detection.process(frame)
            
            if detection_results.detections:
                face_results['face_detected'] = True
                face_results['face_confidence'] = detection_results.detections[0].score[0]
                
                # Face mesh analysis
                mesh_results = self.face_mesh.process(frame)
                
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0]
                    
                    # Extract facial features
                    facial_features = self.extract_facial_features(landmarks, frame.shape)
                    face_results['facial_landmarks'] = facial_features
                    
                    # Emotion recognition
                    emotion_analysis = self.recognize_facial_emotion(frame, landmarks)
                    face_results.update(emotion_analysis)
                    
                    # Micro-expression analysis
                    micro_expressions = self.detect_micro_expressions(facial_features)
                    face_results['micro_expressions'] = micro_expressions
                    
                    # Gaze analysis
                    gaze_analysis = self.analyze_gaze_direction(facial_features)
                    face_results.update(gaze_analysis)
            
        except Exception as e:
            logging.error(f"Error in face analysis: {e}")
        
        return face_results
    
    def extract_facial_features(self, landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Extract specific facial features from landmarks"""
        
        h, w = frame_shape[:2]
        features = {}
        
        try:
            # Convert landmarks to pixel coordinates
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
            
            # Extract specific facial features
            for feature_name, indices in self.landmark_indices.items():
                feature_points = [landmark_points[i] for i in indices if i < len(landmark_points)]
                features[feature_name] = {
                    'points': feature_points,
                    'center': self.calculate_center_point(feature_points),
                    'area': self.calculate_polygon_area(feature_points) if len(feature_points) > 2 else 0
                }
            
            # Calculate derived features
            features['eye_aspect_ratio'] = self.calculate_eye_aspect_ratio(features)
            features['mouth_aspect_ratio'] = self.calculate_mouth_aspect_ratio(features)
            features['eyebrow_height'] = self.calculate_eyebrow_height(features)
            
        except Exception as e:
            logging.error(f"Error extracting facial features: {e}")
        
        return features
    
    def calculate_center_point(self, points: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate center point of a set of points"""
        if not points:
            return (0.0, 0.0)
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    def calculate_polygon_area(self, points: List[Tuple[int, int]]) -> float:
        """Calculate area of polygon using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def calculate_eye_aspect_ratio(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate eye aspect ratio (EAR) for blink detection"""
        
        ear_values = {}
        
        try:
            for eye in ['left_eye', 'right_eye']:
                if eye in features and features[eye]['points']:
                    points = features[eye]['points']
                    
                    if len(points) >= 6:
                        # Simplified EAR calculation
                        # Vertical distances
                        v1 = self.euclidean_distance(points[1], points[5])
                        v2 = self.euclidean_distance(points[2], points[4])
                        
                        # Horizontal distance
                        h = self.euclidean_distance(points[0], points[3])
                        
                        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
                        ear_values[eye] = ear
            
            # Average EAR
            if ear_values:
                ear_values['average'] = sum(ear_values.values()) / len(ear_values)
            
        except Exception as e:
            logging.error(f"Error calculating EAR: {e}")
        
        return ear_values
    
    def calculate_mouth_aspect_ratio(self, features: Dict[str, Any]) -> float:
        """Calculate mouth aspect ratio (MAR) for speech detection"""
        
        try:
            if 'mouth_outer' in features and features['mouth_outer']['points']:
                points = features['mouth_outer']['points']
                
                if len(points) >= 6:
                    # Vertical distance
                    v = self.euclidean_distance(points[2], points[10])
                    
                    # Horizontal distance
                    h = self.euclidean_distance(points[0], points[6])
                    
                    mar = v / h if h > 0 else 0
                    return mar
            
        except Exception as e:
            logging.error(f"Error calculating MAR: {e}")
        
        return 0.0
    
    def calculate_eyebrow_height(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate eyebrow height relative to eyes"""
        
        heights = {}
        
        try:
            for side in ['left', 'right']:
                eyebrow_key = f'{side}_eyebrow'
                eye_key = f'{side}_eye'
                
                if eyebrow_key in features and eye_key in features:
                    eyebrow_center = features[eyebrow_key]['center']
                    eye_center = features[eye_key]['center']
                    
                    height = abs(eyebrow_center[1] - eye_center[1])
                    heights[f'{side}_eyebrow_height'] = height
            
            if heights:
                heights['average_eyebrow_height'] = sum(heights.values()) / len(heights)
        
        except Exception as e:
            logging.error(f"Error calculating eyebrow height: {e}")
        
        return heights
    
    def euclidean_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def recognize_facial_emotion(self, frame: np.ndarray, landmarks) -> Dict[str, Any]:
        """Recognize facial emotion using available models"""
        
        emotion_result = {
            'facial_emotion': 'neutral',
            'emotion_confidence': 0.5,
            'emotion_scores': {'neutral': 1.0},
            'emotion_method': 'fallback'
        }
        
        try:
            # Try transformer models first
            if self.emotion_models:
                for model_name, model in self.emotion_models.items():
                    try:
                        # Convert frame to PIL Image for transformer
                        pil_image = Image.fromarray(frame)
                        
                        # Get predictions
                        predictions = model(pil_image)
                        
                        # Process results
                        emotion_scores = {}
                        for pred in predictions:
                            if isinstance(pred, dict):
                                emotion = self.emotion_mappings.get(pred['label'], pred['label'])
                                emotion_scores[emotion] = pred['score']
                            else:
        # If pred is a list/tuple, access by index
                                emotion = self.emotion_mappings.get(pred[0], pred[0])  # First element is label
                                emotion_scores[emotion] = pred[1]  # Second element is score
                        
                        # Find dominant emotion
                        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                        
                        emotion_result.update({
                            'facial_emotion': dominant_emotion[0],
                            'emotion_confidence': dominant_emotion[1],
                            'emotion_scores': emotion_scores,
                            'emotion_method': f'transformer_{model_name}'
                        })
                        
                        return emotion_result
                        
                    except Exception as e:
                        logging.warning(f"Transformer emotion model {model_name} failed: {e}")
                        continue
            
            # Fallback to MediaPipe-based emotion recognition
            emotion_result = self.analyze_emotion_from_landmarks(landmarks, frame.shape)
            
        except Exception as e:
            logging.error(f"Error in emotion recognition: {e}")
        
        return emotion_result
    
    def analyze_emotion_from_landmarks(self, landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Analyze emotion from facial landmarks using rule-based approach"""
        
        try:
            # Extract facial features
            features = self.extract_facial_features(landmarks, frame_shape)
            
            # Rule-based emotion classification
            emotion_scores = {
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'surprised': 0.0,
                'afraid': 0.0,
                'disgusted': 0.0,
                'neutral': 0.3  # Base score
            }
            
            # Happiness indicators
            if features.get('mouth_aspect_ratio', 0) > 0.05:  # Mouth open/smiling
                emotion_scores['happy'] += 0.4
            
            # Eyebrow position for various emotions
            avg_eyebrow_height = features.get('eyebrow_height', {}).get('average_eyebrow_height', 0)
            if avg_eyebrow_height > 20:  # Raised eyebrows
                emotion_scores['surprised'] += 0.3
                emotion_scores['afraid'] += 0.2
            elif avg_eyebrow_height < 10:  # Lowered eyebrows
                emotion_scores['angry'] += 0.3
                emotion_scores['sad'] += 0.2
            
            # Eye aspect ratio for various emotions
            avg_ear = features.get('eye_aspect_ratio', {}).get('average', 0.3)
            if avg_ear < 0.2:  # Squinted eyes
                emotion_scores['angry'] += 0.2
                emotion_scores['disgusted'] += 0.2
            elif avg_ear > 0.4:  # Wide eyes
                emotion_scores['surprised'] += 0.3
                emotion_scores['afraid'] += 0.3
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            return {
                'facial_emotion': dominant_emotion[0],
                'emotion_confidence': dominant_emotion[1],
                'emotion_scores': emotion_scores,
                'emotion_method': 'mediapipe_landmarks'
            }
            
        except Exception as e:
            logging.error(f"Error in landmark-based emotion analysis: {e}")
            return {
                'facial_emotion': 'neutral',
                'emotion_confidence': 0.3,
                'emotion_scores': {'neutral': 1.0},
                'emotion_method': 'fallback'
            }
    
    def detect_micro_expressions(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect micro-expressions from facial features"""
        
        micro_expressions = []
        
        try:
            # Eye blink detection
            avg_ear = features.get('eye_aspect_ratio', {}).get('average', 0.3)
            if avg_ear < 0.15:
                micro_expressions.append({
                    'type': 'eye_closure',
                    'confidence': 0.7,
                    'description': 'Potential blink or eye strain'
                })
            
            # Mouth movements
            mar = features.get('mouth_aspect_ratio', 0)
            if mar > 0.08:
                micro_expressions.append({
                    'type': 'mouth_opening',
                    'confidence': 0.6,
                    'description': 'Mouth opening detected'
                })
            
            # Eyebrow movements
            eyebrow_heights = features.get('eyebrow_height', {})
            if eyebrow_heights.get('average_eyebrow_height', 0) > 25:
                micro_expressions.append({
                    'type': 'eyebrow_raise',
                    'confidence': 0.8,
                    'description': 'Eyebrow raising detected'
                })
            
            # Asymmetrical expressions
            left_eyebrow = eyebrow_heights.get('left_eyebrow_height', 0)
            right_eyebrow = eyebrow_heights.get('right_eyebrow_height', 0)
            
            if abs(left_eyebrow - right_eyebrow) > 10:
                micro_expressions.append({
                    'type': 'asymmetrical_expression',
                    'confidence': 0.6,
                    'description': 'Asymmetrical facial expression'
                })
        
        except Exception as e:
            logging.error(f"Error detecting micro-expressions: {e}")
        
        return micro_expressions
    
    def analyze_gaze_direction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gaze direction and attention"""
        
        gaze_analysis = {
            'gaze_direction': 'center',
            'gaze_confidence': 0.5,
            'attention_level': 0.5,
            'eye_contact_detected': False
        }
        
        try:
            # Simple gaze estimation based on eye landmarks
            left_eye_center = features.get('left_eye', {}).get('center', (0, 0))
            right_eye_center = features.get('right_eye', {}).get('center', (0, 0))
            
            if left_eye_center != (0, 0) and right_eye_center != (0, 0):
                # Calculate average eye position
                avg_x = (left_eye_center[0] + right_eye_center[0]) / 2
                avg_y = (left_eye_center[1] + right_eye_center[1]) / 2
                
                # Rough gaze direction classification
                # (This is a simplified approach - proper gaze estimation requires eye tracking)
                face_center = features.get('face_oval', {}).get('center', (0, 0))
                
                if face_center != (0, 0):
                    x_offset = avg_x - face_center[0]
                    y_offset = avg_y - face_center[1]
                    
                    # Classify gaze direction
                    if abs(x_offset) < 10 and abs(y_offset) < 10:
                        gaze_analysis['gaze_direction'] = 'center'
                        gaze_analysis['eye_contact_detected'] = True
                        gaze_analysis['attention_level'] = 0.8
                    elif x_offset > 10:
                        gaze_analysis['gaze_direction'] = 'right'
                        gaze_analysis['attention_level'] = 0.4
                    elif x_offset < -10:
                        gaze_analysis['gaze_direction'] = 'left'
                        gaze_analysis['attention_level'] = 0.4
                    elif y_offset > 10:
                        gaze_analysis['gaze_direction'] = 'down'
                        gaze_analysis['attention_level'] = 0.3
                    elif y_offset < -10:
                        gaze_analysis['gaze_direction'] = 'up'
                        gaze_analysis['attention_level'] = 0.6
                    
                    gaze_analysis['gaze_confidence'] = 0.6
        
        except Exception as e:
            logging.error(f"Error in gaze analysis: {e}")
        
        return gaze_analysis
    
    def analyze_hands(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze hand gestures and positions"""
        
        hand_analysis = {
            'hands_detected': 0,
            'hand_positions': [],
            'gestures': [],
            'hand_confidence': 0.0
        }
        
        try:
            results = self.hands.process(frame)
            
            if results.multi_hand_landmarks:
                hand_analysis['hands_detected'] = len(results.multi_hand_landmarks)
                
                confidences = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand classification (left/right)
                    handedness = 'unknown'
                    if results.multi_handedness:
                        handedness = results.multi_handedness[i].classification[0].label
                    
                    # Extract hand position
                    hand_position = self.extract_hand_position(hand_landmarks, frame.shape)
                    hand_analysis['hand_positions'].append({
                        'hand_id': i,
                        'handedness': handedness,
                        'position': hand_position,
                        'landmarks': [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    })
                    
                    # Detect basic gestures
                    gestures = self.detect_hand_gestures(hand_landmarks)
                    hand_analysis['gestures'].extend(gestures)
                    
                    # Confidence estimation
                    if results.multi_handedness:
                        confidences.append(results.multi_handedness[i].classification[0].score)
                
                if confidences:
                    hand_analysis['hand_confidence'] = sum(confidences) / len(confidences)
        
        except Exception as e:
            logging.error(f"Error in hand analysis: {e}")
        
        return hand_analysis
    
    def extract_hand_position(self, hand_landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """Extract hand position information"""
        
        h, w = frame_shape[:2]
        
        # Calculate bounding box
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        return {
            'center_x': sum(x_coords) / len(x_coords),
            'center_y': sum(y_coords) / len(y_coords),
            'bbox': {
                'min_x': min(x_coords),
                'max_x': max(x_coords),
                'min_y': min(y_coords),
                'max_y': max(y_coords)
            }
        }
    
    def detect_hand_gestures(self, hand_landmarks) -> List[Dict[str, Any]]:
        """Detect basic hand gestures"""
        
        gestures = []
        
        try:
            # Get landmark positions
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            # Simple gesture detection based on finger positions
            # Thumb up detection (simplified)
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            index_tip = landmarks[8]
            
            if thumb_tip[1] < thumb_ip[1]:  # Thumb pointing up
                gestures.append({
                    'gesture': 'thumb_up',
                    'confidence': 0.6,
                    'description': 'Thumbs up gesture detected'
                })
            
            # Pointing gesture (index finger extended)
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            
            if (index_tip[1] < index_pip[1] and  # Index extended
                middle_tip[1] > landmarks[10][1]):  # Middle finger folded
                gestures.append({
                    'gesture': 'pointing',
                    'confidence': 0.5,
                    'description': 'Pointing gesture detected'
                })
            
            # Open palm (all fingers extended)
            finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # All fingertips
            finger_pips = [landmarks[i] for i in [3, 6, 10, 14, 18]]  # PIP joints
            
            extended_fingers = sum(1 for tip, pip in zip(finger_tips, finger_pips) if tip[1] < pip[1])
            
            if extended_fingers >= 4:
                gestures.append({
                    'gesture': 'open_palm',
                    'confidence': 0.7,
                    'description': 'Open palm gesture detected'
                })
            elif extended_fingers <= 1:
                gestures.append({
                    'gesture': 'closed_fist',
                    'confidence': 0.7,
                    'description': 'Closed fist detected'
                })
        
        except Exception as e:
            logging.error(f"Error detecting hand gestures: {e}")
        
        return gestures
    
    def analyze_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze body pose and posture"""
        
        pose_analysis = {
            'pose_detected': False,
            'pose_confidence': 0.0,
            'posture': 'unknown',
            'body_language': [],
            'pose_landmarks': []
        }
        
        try:
            results = self.pose.process(frame)
            
            if results.pose_landmarks:
                pose_analysis['pose_detected'] = True
                pose_analysis['pose_landmarks'] = [
                    (lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark
                ]
                
                # Analyze posture
                posture_analysis = self.analyze_posture(results.pose_landmarks.landmark)
                pose_analysis.update(posture_analysis)
                
                # Detect body language cues
                body_language = self.detect_body_language(results.pose_landmarks.landmark)
                pose_analysis['body_language'] = body_language
                
                # Calculate overall pose confidence
                visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
                pose_analysis['pose_confidence'] = sum(visibilities) / len(visibilities)
        
        except Exception as e:
            logging.error(f"Error in pose analysis: {e}")
        
        return pose_analysis
    
    def analyze_posture(self, landmarks) -> Dict[str, Any]:
        """Analyze body posture from pose landmarks"""
        
        posture_analysis = {
            'posture': 'neutral',
            'posture_confidence': 0.5,
            'shoulder_alignment': 'normal',
            'head_position': 'normal'
        }
        
        try:
            # Key pose landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            nose = landmarks[0]
            
            # Check shoulder alignment
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            if shoulder_diff > 0.05:
                posture_analysis['shoulder_alignment'] = 'tilted'
            
            # Check head position relative to shoulders
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            head_forward = nose.y - shoulder_mid_y
            
            if head_forward > 0.1:
                posture_analysis['head_position'] = 'forward'
                posture_analysis['posture'] = 'slouched'
            elif head_forward < -0.05:
                posture_analysis['head_position'] = 'upright'
                posture_analysis['posture'] = 'upright'
            
            # Overall posture classification
            if posture_analysis['shoulder_alignment'] == 'tilted':
                posture_analysis['posture'] = 'asymmetrical'
            
            posture_analysis['posture_confidence'] = 0.7
        
        except Exception as e:
            logging.error(f"Error analyzing posture: {e}")
        
        return posture_analysis
    
    def detect_body_language(self, landmarks) -> List[Dict[str, Any]]:
        """Detect body language cues from pose"""
        
        body_language_cues = []
        
        try:
            # Crossed arms detection (simplified)
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Check if wrists are crossed in front of body
            if (left_wrist.x > right_wrist.x and 
                left_wrist.y < left_shoulder.y and 
                right_wrist.y < right_shoulder.y):
                body_language_cues.append({
                    'cue': 'crossed_arms',
                    'confidence': 0.6,
                    'interpretation': 'Defensive or closed posture'
                })
            
            # Slouched posture detection
            nose = landmarks[0]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            if nose.y > hip_mid_y + 0.2:  # Head significantly forward
                body_language_cues.append({
                    'cue': 'forward_head_posture',
                    'confidence': 0.7,
                    'interpretation': 'Possible fatigue or disengagement'
                })
            
            # Hands near face (self-soothing behavior)
            left_hand = landmarks[19]  # Left pinky
            right_hand = landmarks[20]  # Right pinky
            
            face_region_y = nose.y - 0.1
            
            if (left_hand.y < face_region_y or right_hand.y < face_region_y):
                body_language_cues.append({
                    'cue': 'hand_to_face',
                    'confidence': 0.5,
                    'interpretation': 'Possible self-soothing or anxiety'
                })
        
        except Exception as e:
            logging.error(f"Error detecting body language: {e}")
        
        return body_language_cues
    
    def estimate_engagement(self, frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate overall engagement from visual cues"""
        
        engagement_factors = []
        engagement_score = 0.5  # Base score
        
        # Face detection factor
        if frame_analysis.get('face_detected', False):
            engagement_factors.append(('face_presence', 0.8))
            face_confidence = frame_analysis.get('face_confidence', 0.5)
            engagement_score += face_confidence * 0.2
        
        # Eye contact factor
        if frame_analysis.get('eye_contact_detected', False):
            engagement_factors.append(('eye_contact', 0.9))
            engagement_score += 0.3
        
        # Attention level from gaze
        attention_level = frame_analysis.get('attention_level', 0.5)
        engagement_factors.append(('attention', attention_level))
        engagement_score += (attention_level - 0.5) * 0.3
        
        # Posture factor
        if frame_analysis.get('pose_detected', False):
            posture = frame_analysis.get('posture', 'neutral')
            if posture == 'upright':
                engagement_factors.append(('posture', 0.8))
                engagement_score += 0.1
            elif posture == 'slouched':
                engagement_factors.append(('posture', 0.3))
                engagement_score -= 0.1
        
        # Hand gestures factor
        if frame_analysis.get('hands_detected', 0) > 0:
            gestures = frame_analysis.get('gestures', [])
            if gestures:
                engagement_factors.append(('gestures', 0.7))
                engagement_score += 0.1
        
        # Normalize engagement score
        engagement_score = max(0.0, min(1.0, engagement_score))
        
        # Categorize engagement level
        if engagement_score > 0.7:
            engagement_level = 'high'
        elif engagement_score > 0.4:
            engagement_level = 'medium'
        else:
            engagement_level = 'low'
        
        return {
            'engagement_score': engagement_score,
            'engagement_level': engagement_level,
            'engagement_factors': engagement_factors
        }
    
    def aggregate_frame_analyses(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate analysis results across multiple frames"""
        
        if not frame_analyses:
            return self.empty_analysis()
        
        # Aggregate emotions
        all_emotions = [fa.get('facial_emotion', 'neutral') for fa in frame_analyses]
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        # Average confidence scores
        emotion_confidences = [fa.get('emotion_confidence', 0.0) for fa in frame_analyses if 'emotion_confidence' in fa]
        avg_emotion_confidence = sum(emotion_confidences) / len(emotion_confidences) if emotion_confidences else 0.0
        
        # Average engagement
        engagement_scores = [fa.get('engagement_score', 0.5) for fa in frame_analyses if 'engagement_score' in fa]
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.5
        
        # Face detection rate
        face_detections = sum(1 for fa in frame_analyses if fa.get('face_detected', False))
        face_detection_rate = face_detections / len(frame_analyses)
        
        # Aggregate micro-expressions
        all_micro_expressions = []
        for fa in frame_analyses:
            all_micro_expressions.extend(fa.get('micro_expressions', []))
        
        # Count gesture occurrences
        all_gestures = []
        for fa in frame_analyses:
            all_gestures.extend(fa.get('gestures', []))
        
        gesture_counts = {}
        for gesture in all_gestures:
            gesture_type = gesture.get('gesture', 'unknown')
            gesture_counts[gesture_type] = gesture_counts.get(gesture_type, 0) + 1
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_confidence': avg_emotion_confidence,
            'emotion_distribution': emotion_counts,
            'engagement_score': avg_engagement,
            'face_detection_rate': face_detection_rate,
            'micro_expressions': all_micro_expressions,
            'gesture_counts': gesture_counts,
            'analysis_method': 'frame_aggregation'
        }
    
    def analyze_temporal_patterns(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns across frames"""
        
        temporal_analysis = {
            'emotion_stability': 0.5,
            'engagement_trend': 'stable',
            'temporal_patterns': []
        }
        
        try:
            if len(frame_analyses) < 2:
                return temporal_analysis
            
            # Analyze emotion stability
            emotions = [fa.get('facial_emotion', 'neutral') for fa in frame_analyses]
            emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
            emotion_stability = 1.0 - (emotion_changes / (len(emotions) - 1))
            temporal_analysis['emotion_stability'] = emotion_stability
            
            # Analyze engagement trend
            engagement_scores = [fa.get('engagement_score', 0.5) for fa in frame_analyses]
            if len(engagement_scores) >= 3:
                early_avg = sum(engagement_scores[:len(engagement_scores)//2]) / (len(engagement_scores)//2)
                late_avg = sum(engagement_scores[len(engagement_scores)//2:]) / (len(engagement_scores) - len(engagement_scores)//2)
                
                if late_avg > early_avg + 0.1:
                    temporal_analysis['engagement_trend'] = 'increasing'
                elif late_avg < early_avg - 0.1:
                    temporal_analysis['engagement_trend'] = 'decreasing'
                else:
                    temporal_analysis['engagement_trend'] = 'stable'
            
            # Detect temporal patterns
            patterns = []
            
            # Blinking pattern
            blink_frames = [i for i, fa in enumerate(frame_analyses) 
                           if any(me.get('type') == 'eye_closure' for me in fa.get('micro_expressions', []))]
            
            if len(blink_frames) > 1:
                blink_intervals = [blink_frames[i+1] - blink_frames[i] for i in range(len(blink_frames)-1)]
                avg_blink_interval = sum(blink_intervals) / len(blink_intervals)
                
                patterns.append({
                    'pattern': 'blink_rate',
                    'value': len(blink_frames) / len(frame_analyses),
                    'interpretation': 'Normal' if 10 < avg_blink_interval < 60 else 'Unusual blink rate'
                })
            
            temporal_analysis['temporal_patterns'] = patterns
        
        except Exception as e:
            logging.error(f"Error in temporal analysis: {e}")
        
        return temporal_analysis
    
    def compute_overall_assessment(self, aggregated_results: Dict[str, Any], 
                                 temporal_analysis: Dict[str, Any], 
                                 total_frames: int) -> Dict[str, Any]:
        """Compute overall visual analysis assessment"""
        
        overall_assessment = {
            'analysis_quality': 'poor',
            'confidence': 0.0,
            'recommendations': []
        }
        
        try:
            # Calculate analysis quality
            quality_factors = []
            
            # Face detection quality
            face_detection_rate = aggregated_results.get('face_detection_rate', 0.0)
            if face_detection_rate > 0.8:
                quality_factors.append(0.9)
            elif face_detection_rate > 0.5:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.3)
            
            # Number of frames factor
            if total_frames >= 10:
                quality_factors.append(0.8)
            elif total_frames >= 5:
                quality_factors.append(0.6)
            else:
                quality_factors.append(0.4)
            
            # Emotion confidence factor
            emotion_confidence = aggregated_results.get('emotion_confidence', 0.0)
            quality_factors.append(emotion_confidence)
            
            # Overall quality score
            quality_score = sum(quality_factors) / len(quality_factors)
            
            if quality_score > 0.8:
                overall_assessment['analysis_quality'] = 'high'
            elif quality_score > 0.6:
                overall_assessment['analysis_quality'] = 'medium'
            else:
                overall_assessment['analysis_quality'] = 'low'
            
            overall_assessment['confidence'] = quality_score
            
            # Generate recommendations
            recommendations = []
            
            dominant_emotion = aggregated_results.get('dominant_emotion', 'neutral')
            if dominant_emotion in ['sad', 'angry', 'afraid']:
                recommendations.append(f"Detected {dominant_emotion} emotion - consider supportive intervention")
            
            engagement_score = aggregated_results.get('engagement_score', 0.5)
            if engagement_score < 0.4:
                recommendations.append("Low engagement detected - may need attention or break")
            
            emotion_stability = temporal_analysis.get('emotion_stability', 0.5)
            if emotion_stability < 0.3:
                recommendations.append("High emotional variability - monitor for distress")
            
            if face_detection_rate < 0.5:
                recommendations.append("Poor face detection - consider improving video quality")
            
            overall_assessment['recommendations'] = recommendations
        
        except Exception as e:
            logging.error(f"Error computing overall assessment: {e}")
        
        return overall_assessment
    
    def empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'face_detected': False,
            'dominant_emotion': 'neutral',
            'emotion_confidence': 0.0,
            'engagement_score': 0.0,
            'engagement_level': 'unknown',
            'analysis_quality': 'poor',
            'confidence': 0.0,
            'error': 'No visual data to analyze'
        }
    
    def fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when main analysis fails"""
        return {
            'face_detected': False,
            'dominant_emotion': 'neutral',
            'emotion_confidence': 0.3,
            'engagement_score': 0.5,
            'engagement_level': 'unknown',
            'analysis_quality': 'poor',
            'confidence': 0.2,
            'error': 'Visual analysis failed - using fallback'
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the visual analyzer
    analyzer = VisualAnalyzer()
    
    print("Visual Analyzer Test")
    print("=" * 50)
    
    # Create a simple test image (since we don't have real images)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Testing with synthetic image...")
    results = analyzer.analyze_image(test_image)
    
    print(f"Face Detected: {results['face_detected']}")
    print(f"Dominant Emotion: {results['dominant_emotion']}")
    print(f"Emotion Confidence: {results['emotion_confidence']:.2f}")
    print(f"Engagement Score: {results['engagement_score']:.2f}")
    print(f"Analysis Quality: {results['analysis_quality']}")
    print(f"Overall Confidence: {results['confidence']:.2f}")
    
    if results.get('recommendations'):
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"• {rec}")
    
    print(f"\nAnalyzer Components Loaded:")
    print(f"• MediaPipe Face Detection: ✓")
    print(f"• MediaPipe Face Mesh: ✓") 
    print(f"• MediaPipe Hands: ✓")
    print(f"• MediaPipe Pose: ✓")
    print(f"• Transformer Models: {len(analyzer.emotion_models)} available")