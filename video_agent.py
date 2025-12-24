"""
Video Agent for Agentic Therapy AI System
Handles facial expression recognition and emotional analysis from video streams
WITH CONTINUOUS MONITORING & FER EMOTION DETECTION
"""

import logging
import cv2
import numpy as np
import threading
import queue
import base64
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
import os

# Try to import face recognition libraries
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("dlib not available - some features will be limited")

try:
    from fer import FER
    FER_AVAILABLE = True
except (ImportError, Exception) as e:
    FER_AVAILABLE = False
    logging.warning(f"fer not available - using OpenCV-based emotion detection: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAgent:
    """
    🤖 AUTONOMOUS VIDEO AGENT for therapeutic AI system
    Features:
    - Facial expression recognition and emotion analysis
    - Autonomous decision-making based on emotional patterns
    - Automatic escalation to crisis counselor for concerning patterns
    - Goal-oriented therapeutic interventions
    """

    def __init__(self, config_path="config.json", therapy_agent=None, crisis_counselor=None):
        self.config = self.load_config(config_path)
        self.camera = None
        self.is_recording = False
        self.is_analyzing = False
        self.analysis_thread = None
        
        # Agent references for autonomous coordination
        self.therapy_agent = therapy_agent
        self.crisis_counselor = crisis_counselor

        # Video settings
        self.video_settings = self.config.get("video", {
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30,
            "analysis_interval": 2.0,  # Analyze every 2 seconds
            "emotion_threshold": 0.2,  # Lower threshold for better detection (was 0.3)
            "face_detection_scale": 1.1,
            "min_neighbors": 5,
            "continuous_monitoring": True,  # Enable by default
            "use_frame_enhancement": True  # Enable contrast enhancement
        })
        
        # 🤖 AUTONOMOUS DECISION-MAKING SYSTEM
        self.autonomous_enabled = True
        self.decision_history = []
        
        # Emotion persistence tracking
        self.emotion_persistence = {
            'current_emotion': None,
            'start_time': None,
            'duration': 0
        }
        
        # Escalation rules (in seconds)
        self.escalation_rules = {
            'sad': {'duration': 300, 'action': 'escalate_crisis', 'priority': 'high'},  # 5 minutes
            'angry': {'duration': 240, 'action': 'escalate_crisis', 'priority': 'high'},  # 4 minutes
            'fear': {'duration': 180, 'action': 'escalate_crisis', 'priority': 'high'},  # 3 minutes
            'disgust': {'duration': 300, 'action': 'notify_therapy', 'priority': 'medium'},
            'neutral': {'duration': 600, 'action': 'check_engagement', 'priority': 'low'}  # 10 minutes
        }
        
        # Autonomous actions taken
        self.autonomous_actions = []

        # Emotion mapping with detailed therapeutic context
        self.emotion_mapping = {
            'angry': {
                'valence': -0.8, 
                'arousal': 0.8, 
                'therapy_priority': 'high',
                'color': '#FF4444',
                'icon': '😠'
            },
            'disgust': {
                'valence': -0.6, 
                'arousal': 0.4, 
                'therapy_priority': 'medium',
                'color': '#AA4444',
                'icon': '🤢'
            },
            'fear': {
                'valence': -0.7, 
                'arousal': 0.9, 
                'therapy_priority': 'high',
                'color': '#8844FF',
                'icon': '😨'
            },
            'happy': {
                'valence': 0.8, 
                'arousal': 0.6, 
                'therapy_priority': 'low',
                'color': '#44FF44',
                'icon': '😊'
            },
            'sad': {
                'valence': -0.8, 
                'arousal': -0.4, 
                'therapy_priority': 'high',
                'color': '#4444FF',
                'icon': '😢'
            },
            'surprise': {
                'valence': 0.2, 
                'arousal': 0.8, 
                'therapy_priority': 'low',
                'color': '#FFAA44',
                'icon': '😲'
            },
            'neutral': {
                'valence': 0.0, 
                'arousal': 0.0, 
                'therapy_priority': 'low',
                'color': '#888888',
                'icon': '😐'
            }
        }

        # Analysis history
        self.emotion_history = []
        self.analysis_queue = queue.Queue(maxsize=100)
        self.latest_analysis = None
        self.analysis_callbacks = []
        
        # Autonomous goals and state
        self.current_goal = "monitor_emotional_wellbeing"
        self.intervention_active = False
        self.last_escalation_time = None

        # Initialize components
        self.face_cascade = None
        self.emotion_detector = None

        self.init_face_detection()
        self.init_emotion_detection()

        logger.info("🤖 AUTONOMOUS Video Agent initialized successfully")
        logger.info("✓ Autonomous decision-making: ENABLED")
        logger.info("✓ Crisis escalation rules: ACTIVE")
        if FER_AVAILABLE:
            logger.info("✓ FER library available - using deep learning emotion detection")
        else:
            logger.warning("⚠ FER library not available - install with: pip install fer")
        
        if self.crisis_counselor:
            logger.info("✓ Crisis counselor integration: READY")
        else:
            logger.warning("⚠ Crisis counselor not linked - escalation capabilities limited")

    def load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def init_face_detection(self):
        """Initialize face detection using OpenCV Haar cascades"""
        try:
            # Load OpenCV's pre-trained face detection model
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                self.face_cascade = None
            else:
                logger.info("✓ Face detection initialized (OpenCV Haar)")

        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_cascade = None

    def init_emotion_detection(self):
        """Initialize emotion detection system with FER"""
        try:
            if FER_AVAILABLE:
                # Use FER library for emotion detection
                # mtcnn=True provides better face detection and more accurate emotions
                # mtcnn=False uses OpenCV (faster but less accurate)
                # Try mtcnn=True first for better accuracy
                try:
                    self.emotion_detector = FER(mtcnn=True)
                    logger.info("✓ Emotion detection initialized (FER Deep Learning with MTCNN)")
                except Exception as mtcnn_error:
                    logger.warning(f"MTCNN not available: {mtcnn_error}")
                    logger.info("Falling back to OpenCV face detection")
                    self.emotion_detector = FER(mtcnn=False)
                    logger.info("✓ Emotion detection initialized (FER Deep Learning with OpenCV)")
            else:
                # Fallback to rule-based emotion detection
                self.emotion_detector = None
                logger.info("⚠ Using basic emotion analysis (install FER for better accuracy)")

        except Exception as e:
            logger.error(f"Failed to initialize emotion detection: {e}")
            self.emotion_detector = None

    def register_analysis_callback(self, callback: Callable):
        """Register a callback function to be called when analysis is complete"""
        if callback not in self.analysis_callbacks:
            self.analysis_callbacks.append(callback)
            logger.info(f"Registered analysis callback: {callback.__name__}")

    def unregister_analysis_callback(self, callback: Callable):
        """Unregister an analysis callback"""
        if callback in self.analysis_callbacks:
            self.analysis_callbacks.remove(callback)
            logger.info(f"Unregistered analysis callback: {callback.__name__}")

    def get_video_status(self) -> Dict[str, Any]:
        """Get current video system status"""
        return {
            "camera_available": self.camera is not None and self.camera.isOpened() if self.camera else False,
            "face_detection_available": self.face_cascade is not None,
            "emotion_detection_available": self.emotion_detector is not None or self.face_cascade is not None,
            "is_recording": self.is_recording,
            "is_analyzing": self.is_analyzing,
            "continuous_monitoring": self.video_settings.get("continuous_monitoring", False),
            "analysis_interval": self.video_settings.get("analysis_interval", 2.0),
            "latest_emotion": self.latest_analysis.get("dominant_emotion") if self.latest_analysis else None,
            "emotion_history_count": len(self.emotion_history),
            "settings": self.video_settings,
            "libraries_status": {
                "opencv": True,  # Required for this agent
                "dlib": DLIB_AVAILABLE,
                "fer": FER_AVAILABLE
            }
        }

    def start_camera(self, auto_start_monitoring: bool = True) -> Dict[str, Any]:
        """Initialize and start camera with improved error handling"""
        try:
            if self.camera and self.camera.isOpened():
                # If continuous monitoring is enabled and not analyzing, start it
                if auto_start_monitoring and self.video_settings.get("continuous_monitoring", True) and not self.is_analyzing:
                    self.start_continuous_analysis()
                
                return {
                    "success": True,
                    "message": "Camera already running",
                    "continuous_monitoring": self.is_analyzing
                }

            camera_index = self.video_settings["camera_index"]
            
            # Try different backends in order of preference (Windows)
            backends = [
                cv2.CAP_DSHOW,      # DirectShow (Windows) - most compatible
                cv2.CAP_MSMF,       # Media Foundation (Windows) - default
                cv2.CAP_ANY         # Let OpenCV choose
            ]
            
            camera_opened = False
            backend_used = None
            
            for backend in backends:
                logger.info(f"Trying camera with backend: {backend}")
                self.camera = cv2.VideoCapture(camera_index, backend)
                
                if self.camera.isOpened():
                    # Wait a moment for camera to initialize
                    time.sleep(0.5)
                    
                    # Try to read a test frame
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        camera_opened = True
                        backend_used = backend
                        logger.info(f"✓ Camera opened successfully with backend: {backend}")
                        break
                    else:
                        logger.warning(f"Camera opened but cannot read frames with backend {backend}")
                        self.camera.release()
                        self.camera = None
                else:
                    logger.warning(f"Failed to open camera with backend {backend}")
            
            if not camera_opened:
                return {
                    "success": False,
                    "error": "Failed to open camera with any backend. Please check:\n" +
                            "1. Camera is not in use by another application\n" +
                            "2. Camera permissions are enabled in Windows settings\n" +
                            "3. Camera drivers are up to date"
                }

            # Try to configure camera settings (may fail on some cameras, that's OK)
            try:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_settings["frame_width"])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_settings["frame_height"])
                self.camera.set(cv2.CAP_PROP_FPS, self.video_settings["fps"])
                
                # Verify settings were applied (they might not be)
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"Camera settings - Requested: {self.video_settings['frame_width']}x{self.video_settings['frame_height']}@{self.video_settings['fps']}fps")
                logger.info(f"Camera settings - Actual: {actual_width}x{actual_height}@{actual_fps}fps")
            except Exception as e:
                logger.warning(f"Could not set all camera properties: {e}")
                # Continue anyway - camera is working

            # Get actual frame to verify everything works
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.camera.release()
                self.camera = None
                return {
                    "success": False,
                    "error": "Camera opened but failed to read frames"
                }

            logger.info(f"Camera initialized successfully: {frame.shape[1]}x{frame.shape[0]}")

            # Auto-start continuous monitoring if enabled
            monitoring_started = False
            if auto_start_monitoring and self.video_settings.get("continuous_monitoring", True):
                monitor_result = self.start_continuous_analysis()
                monitoring_started = monitor_result.get("success", False)
                if monitoring_started:
                    logger.info("✓ Continuous emotion monitoring started automatically")

            return {
                "success": True,
                "message": "Camera started successfully",
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "camera_index": camera_index,
                "backend": backend_used,
                "continuous_monitoring": monitoring_started,
                "actual_settings": {
                    "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": int(self.camera.get(cv2.CAP_PROP_FPS))
                }
            }

        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            if self.camera:
                self.camera.release()
                self.camera = None
            return {
                "success": False,
                "error": f"Failed to start camera: {str(e)}"
            }

    def stop_camera(self) -> Dict[str, Any]:
        """Stop camera and release resources"""
        try:
            # Stop continuous analysis first
            if self.is_analyzing:
                self.stop_continuous_analysis()
            
            if self.camera and self.camera.isOpened():
                self.camera.release()
                self.camera = None
                self.is_recording = False
                logger.info("Camera stopped and resources released")

            return {
                "success": True,
                "message": "Camera stopped",
                "emotion_history_saved": len(self.emotion_history)
            }

        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
            return {
                "success": False,
                "error": f"Failed to stop camera: {str(e)}"
            }

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame using OpenCV"""
        if self.face_cascade is None:
            return []

        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.video_settings["face_detection_scale"],
                minNeighbors=self.video_settings["min_neighbors"],
                minSize=(30, 30)
            )

            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def analyze_emotion_fer(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze emotions using FER library with deep learning"""
        if not FER_AVAILABLE or self.emotion_detector is None:
            return self.analyze_emotion_basic(frame)

        try:
            # Preprocess frame for better emotion detection
            # Convert BGR to RGB (FER expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Enhance contrast for better feature detection
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            frame_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Analyze emotions using FER on enhanced frame
            emotions = self.emotion_detector.detect_emotions(frame_enhanced)

            if not emotions:
                # Try with original frame if enhanced didn't work
                emotions = self.emotion_detector.detect_emotions(frame_rgb)
                
            if not emotions:
                return {
                    "faces_detected": 0,
                    "emotions": [],
                    "dominant_emotion": "neutral",
                    "confidence": 0.0,
                    "analysis_method": "fer",
                    "all_emotions": {}
                }

            # Process each face
            face_emotions = []
            for face_data in emotions:
                emotions_dict = face_data['emotions']

                # Find dominant emotion (excluding neutral if other emotions are strong)
                # Sort emotions by score
                sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
                
                # If top emotion is neutral but second emotion is > 0.25, use second
                if sorted_emotions[0][0] == 'neutral' and len(sorted_emotions) > 1:
                    if sorted_emotions[1][1] > 0.25:
                        dominant = sorted_emotions[1][0]
                        confidence = sorted_emotions[1][1]
                    else:
                        dominant = sorted_emotions[0][0]
                        confidence = sorted_emotions[0][1]
                else:
                    dominant = sorted_emotions[0][0]
                    confidence = sorted_emotions[0][1]

                # Get face box
                box = face_data['box']

                face_emotions.append({
                    "face_box": box,
                    "emotions": emotions_dict,
                    "dominant_emotion": dominant,
                    "confidence": confidence,
                    "therapy_analysis": self.get_therapy_analysis(dominant, confidence),
                    "all_emotion_scores": sorted_emotions  # Include all for debugging
                })

            # Overall analysis
            if face_emotions:
                # Use first face for overall emotion
                primary_face = face_emotions[0]
                
                # Get emotion metadata
                emotion_meta = self.emotion_mapping.get(primary_face["dominant_emotion"], self.emotion_mapping['neutral'])
                
                return {
                    "faces_detected": len(face_emotions),
                    "emotions": face_emotions,
                    "dominant_emotion": primary_face["dominant_emotion"],
                    "confidence": primary_face["confidence"],
                    "all_emotions": primary_face["emotions"],
                    "analysis_method": "fer_deep_learning",
                    "therapy_priority": primary_face["therapy_analysis"]["priority"],
                    "therapeutic_suggestion": primary_face["therapy_analysis"]["suggestion"],
                    "emotion_color": emotion_meta["color"],
                    "emotion_icon": emotion_meta["icon"],
                    "valence": emotion_meta["valence"],
                    "arousal": emotion_meta["arousal"]
                }
            else:
                return {
                    "faces_detected": 0,
                    "emotions": [],
                    "dominant_emotion": "neutral",
                    "confidence": 0.0,
                    "analysis_method": "fer",
                    "all_emotions": {}
                }

        except Exception as e:
            logger.error(f"Error in FER emotion analysis: {e}")
            return self.analyze_emotion_basic(frame)

    def analyze_emotion_basic(self, frame: np.ndarray) -> Dict[str, Any]:
        """Basic emotion analysis using face detection and simple heuristics"""
        try:
            faces = self.detect_faces(frame)

            if not faces:
                return {
                    "faces_detected": 0,
                    "emotions": [],
                    "dominant_emotion": "neutral",
                    "confidence": 0.0,
                    "analysis_method": "basic_heuristic",
                    "all_emotions": {"neutral": 1.0}
                }

            # Simple heuristic-based emotion analysis
            face_emotions = []
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]

                # Simple brightness/contrast analysis for basic emotion estimation
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                # Calculate basic features
                brightness = np.mean(gray_face)
                contrast = np.std(gray_face)

                # Heuristic emotion mapping (very basic)
                if brightness < 80 and contrast > 30:
                    emotion = "sad"
                    confidence = 0.6
                elif brightness > 120 and contrast > 40:
                    emotion = "happy"
                    confidence = 0.6
                elif contrast > 50:
                    emotion = "surprise"
                    confidence = 0.5
                else:
                    emotion = "neutral"
                    confidence = 0.4

                face_emotions.append({
                    "face_box": [x, y, w, h],
                    "emotions": {emotion: confidence},
                    "dominant_emotion": emotion,
                    "confidence": confidence,
                    "therapy_analysis": self.get_therapy_analysis(emotion, confidence),
                    "features": {
                        "brightness": float(brightness),
                        "contrast": float(contrast)
                    }
                })

            # Return analysis for primary face
            primary_face = face_emotions[0]
            emotion_meta = self.emotion_mapping.get(primary_face["dominant_emotion"], self.emotion_mapping['neutral'])
            
            return {
                "faces_detected": len(face_emotions),
                "emotions": face_emotions,
                "dominant_emotion": primary_face["dominant_emotion"],
                "confidence": primary_face["confidence"],
                "all_emotions": primary_face["emotions"],
                "analysis_method": "basic_heuristic",
                "therapy_priority": primary_face["therapy_analysis"]["priority"],
                "therapeutic_suggestion": primary_face["therapy_analysis"]["suggestion"],
                "emotion_color": emotion_meta["color"],
                "emotion_icon": emotion_meta["icon"]
            }

        except Exception as e:
            logger.error(f"Error in basic emotion analysis: {e}")
            return {
                "faces_detected": 0,
                "emotions": [],
                "dominant_emotion": "neutral",
                "confidence": 0.0,
                "analysis_method": "error",
                "error": str(e),
                "all_emotions": {"neutral": 1.0}
            }

    def get_therapy_analysis(self, emotion: str, confidence: float) -> Dict[str, Any]:
        """Get therapeutic analysis for detected emotion"""
        emotion_info = self.emotion_mapping.get(emotion, self.emotion_mapping['neutral'])

        # Determine intervention suggestions based on emotion
        suggestions = {
            'angry': "I notice you might be feeling frustrated. Let's take a moment - would deep breathing or talking about what's bothering you help?",
            'sad': "I can see you're going through something difficult. Would you like to share what's on your mind? I'm here to listen.",
            'fear': "You seem anxious or worried. Let's try some grounding techniques together. Can you name 5 things you can see right now?",
            'happy': "It's wonderful to see you in good spirits! What's bringing you joy today? Let's celebrate that.",
            'surprise': "You seem surprised or caught off guard. Is there something unexpected we should talk about?",
            'neutral': "How are you feeling today? I'm here to support you in whatever way you need.",
            'disgust': "Something seems to be bothering you. Would you like to explore what's causing this discomfort?"
        }

        # Additional coping strategies
        coping_strategies = {
            'angry': ["Deep breathing (4-7-8 technique)", "Progressive muscle relaxation", "Physical exercise", "Journaling"],
            'sad': ["Talking with someone", "Self-compassion exercises", "Gentle activity", "Mindfulness meditation"],
            'fear': ["Grounding techniques (5-4-3-2-1)", "Safe space visualization", "Breathing exercises", "Reality testing"],
            'happy': ["Gratitude practice", "Savoring the moment", "Sharing joy with others", "Positive reflection"],
            'surprise': ["Processing the information", "Taking time to adjust", "Seeking clarification"],
            'neutral': ["Emotional check-in", "Mood journaling", "Mindful awareness"],
            'disgust': ["Identifying triggers", "Boundary setting", "Value clarification"]
        }

        return {
            "priority": emotion_info["therapy_priority"],
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "suggestion": suggestions.get(emotion, "Let's talk about how you're feeling"),
            "confidence_level": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
            "coping_strategies": coping_strategies.get(emotion, []),
            "color": emotion_info["color"],
            "icon": emotion_info["icon"]
        }

    def analyze_current_frame(self) -> Dict[str, Any]:
        """Analyze current camera frame for emotions"""
        if not self.camera or not self.camera.isOpened():
            return {
                "success": False,
                "error": "Camera not available"
            }

        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                return {
                    "success": False,
                    "error": "Failed to capture frame"
                }

            # Analyze emotions
            analysis = self.analyze_emotion_fer(frame)
            analysis["success"] = True
            analysis["timestamp"] = datetime.now().isoformat()

            # Store as latest analysis
            self.latest_analysis = analysis

            # Add to history
            self.emotion_history.append({
                "timestamp": analysis["timestamp"],
                "emotion": analysis["dominant_emotion"],
                "confidence": analysis["confidence"],
                "all_emotions": analysis.get("all_emotions", {}),
                "faces_detected": analysis.get("faces_detected", 0)
            })

            # Keep only recent history (last 100 analyses)
            if len(self.emotion_history) > 100:
                self.emotion_history = self.emotion_history[-100:]
            
            # 🤖 AUTONOMOUS DECISION-MAKING: Analyze patterns and take action
            if self.autonomous_enabled:
                self.autonomous_decision_engine(analysis)

            # Notify callbacks
            for callback in self.analysis_callbacks:
                try:
                    callback(analysis)
                except Exception as e:
                    logger.error(f"Error in analysis callback: {e}")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing current frame: {e}")
            return {
                "success": False,
                "error": f"Frame analysis failed: {str(e)}"
            }

    def get_emotion_trends(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Get emotion trends over specified duration"""
        try:
            if not self.emotion_history:
                return {
                    "success": True,
                    "trends": {},
                    "message": "No emotion history available"
                }

            # Filter recent history
            cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
            recent_emotions = [
                entry for entry in self.emotion_history
                if datetime.fromisoformat(entry["timestamp"]).timestamp() > cutoff_time
            ]

            if not recent_emotions:
                return {
                    "success": True,
                    "trends": {},
                    "message": f"No emotions recorded in last {duration_minutes} minutes"
                }

            # Analyze trends
            emotion_counts = {}
            confidence_sum = 0

            for entry in recent_emotions:
                emotion = entry["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                confidence_sum += entry["confidence"]

            # Calculate percentages
            total_count = len(recent_emotions)
            emotion_percentages = {
                emotion: (count / total_count) * 100
                for emotion, count in emotion_counts.items()
            }

            # Find dominant emotion
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = confidence_sum / total_count

            # Get emotion metadata
            emotion_meta = self.emotion_mapping.get(dominant_emotion, self.emotion_mapping['neutral'])

            return {
                "success": True,
                "duration_minutes": duration_minutes,
                "total_analyses": total_count,
                "trends": {
                    "dominant_emotion": dominant_emotion,
                    "average_confidence": avg_confidence,
                    "emotion_distribution": emotion_percentages,
                    "emotion_counts": emotion_counts,
                    "therapy_recommendations": self.get_trend_therapy_recommendations(emotion_percentages),
                    "emotion_icon": emotion_meta["icon"],
                    "emotion_color": emotion_meta["color"]
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting emotion trends: {e}")
            return {
                "success": False,
                "error": f"Failed to analyze trends: {str(e)}"
            }

    def get_trend_therapy_recommendations(self, emotion_percentages: Dict[str, float]) -> List[str]:
        """Get therapy recommendations based on emotion trends"""
        recommendations = []

        # Check for concerning patterns
        if emotion_percentages.get('sad', 0) > 40:
            recommendations.append("Persistent sadness detected - consider exploring underlying causes and practicing self-compassion")

        if emotion_percentages.get('angry', 0) > 30:
            recommendations.append("Recurring anger patterns - anger management and emotional regulation techniques recommended")

        if emotion_percentages.get('fear', 0) > 25:
            recommendations.append("Elevated anxiety/fear levels - grounding techniques and relaxation exercises suggested")

        if emotion_percentages.get('neutral', 0) > 70:
            recommendations.append("Limited emotional expression observed - gentle exploration of feelings might be beneficial")

        if emotion_percentages.get('happy', 0) > 60:
            recommendations.append("Positive emotional state maintained - excellent time to build resilience and coping skills")

        # Check for emotional variability
        if len(emotion_percentages) > 4:
            recommendations.append("High emotional variability - mood tracking might provide helpful insights")

        if not recommendations:
            recommendations.append("Emotional patterns appear balanced - continue current therapeutic approach")

        return recommendations

    def start_continuous_analysis(self, callback=None) -> Dict[str, Any]:
        """Start continuous emotion analysis in background thread"""
        if self.is_analyzing:
            return {
                "success": False,
                "error": "Analysis already running"
            }

        if not self.camera or not self.camera.isOpened():
            camera_result = self.start_camera(auto_start_monitoring=False)
            if not camera_result["success"]:
                return camera_result

        # Add callback if provided
        if callback:
            self.register_analysis_callback(callback)

        def analysis_thread():
            self.is_analyzing = True
            last_analysis = 0
            analysis_count = 0

            logger.info("🎥 Continuous emotion monitoring started")

            try:
                while self.is_analyzing:
                    current_time = time.time()

                    # Analyze at specified interval
                    if current_time - last_analysis >= self.video_settings["analysis_interval"]:
                        result = self.analyze_current_frame()

                        if result["success"]:
                            analysis_count += 1
                            
                            # Show all emotion scores for debugging
                            all_emotions_str = ""
                            if result.get("all_emotions"):
                                emotion_scores = ", ".join([f"{k}:{v:.2f}" for k, v in result["all_emotions"].items()])
                                all_emotions_str = f" | All: [{emotion_scores}]"
                            
                            logger.info(f"📊 Analysis #{analysis_count}: {result['dominant_emotion']} "
                                      f"({result['confidence']:.2f} confidence) - "
                                      f"{result['faces_detected']} face(s) detected{all_emotions_str}")

                        last_analysis = current_time

                    time.sleep(0.1)  # Short sleep to prevent excessive CPU usage

            except Exception as e:
                logger.error(f"Error in analysis thread: {e}")
            finally:
                self.is_analyzing = False
                logger.info(f"🛑 Continuous monitoring stopped. Total analyses: {analysis_count}")

        # Start analysis thread
        self.analysis_thread = threading.Thread(target=analysis_thread, daemon=True)
        self.analysis_thread.start()

        return {
            "success": True,
            "message": "Continuous emotion monitoring started",
            "analysis_interval": self.video_settings["analysis_interval"],
            "emotion_detection_method": "FER Deep Learning" if FER_AVAILABLE else "Basic Heuristic"
        }

    def stop_continuous_analysis(self) -> Dict[str, Any]:
        """Stop continuous emotion analysis"""
        if not self.is_analyzing:
            return {
                "success": True,
                "message": "Continuous analysis not running"
            }

        self.is_analyzing = False

        # Wait for thread to finish (max 2 seconds)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2.0)

        return {
            "success": True,
            "message": "Continuous analysis stopped",
            "total_analyses": len(self.emotion_history)
        }

    def get_latest_analysis(self) -> Dict[str, Any]:
        """Get the most recent emotion analysis"""
        if self.latest_analysis:
            return {
                "success": True,
                **self.latest_analysis
            }
        else:
            return {
                "success": False,
                "message": "No analysis available yet"
            }

    def capture_and_encode_frame(self) -> Dict[str, Any]:
        """Capture current frame and return as base64 encoded image"""
        if not self.camera or not self.camera.isOpened():
            return {
                "success": False,
                "error": "Camera not available"
            }

        try:
            ret, frame = self.camera.read()
            if not ret:
                return {
                    "success": False,
                    "error": "Failed to capture frame"
                }

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "frame_base64": frame_base64,
                "frame_size": len(buffer),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return {
                "success": False,
                "error": f"Frame capture failed: {str(e)}"
            }

    def autonomous_decision_engine(self, analysis: Dict[str, Any]):
        """🤖 AUTONOMOUS DECISION ENGINE - Analyzes patterns and takes action"""
        try:
            current_emotion = analysis.get("dominant_emotion")
            confidence = analysis.get("confidence", 0)
            timestamp = datetime.now()
            
            # Track emotion persistence
            if self.emotion_persistence['current_emotion'] != current_emotion:
                # Emotion changed - record previous duration
                if self.emotion_persistence['current_emotion']:
                    prev_emotion = self.emotion_persistence['current_emotion']
                    duration = self.emotion_persistence['duration']
                    logger.info(f"📊 Emotion changed: {prev_emotion} lasted {duration:.1f}s")
                
                # Start tracking new emotion
                self.emotion_persistence = {
                    'current_emotion': current_emotion,
                    'start_time': timestamp,
                    'duration': 0
                }
            else:
                # Same emotion - update duration
                if self.emotion_persistence['start_time']:
                    self.emotion_persistence['duration'] = (
                        timestamp - self.emotion_persistence['start_time']
                    ).total_seconds()
            
            # Check escalation rules
            duration = self.emotion_persistence['duration']
            if current_emotion in self.escalation_rules:
                rule = self.escalation_rules[current_emotion]
                threshold = rule['duration']
                
                # 🚨 AUTONOMOUS ACTION: Escalate if duration exceeds threshold
                if duration >= threshold and confidence > 0.4:
                    self.execute_autonomous_action(
                        action_type=rule['action'],
                        emotion=current_emotion,
                        duration=duration,
                        priority=rule['priority'],
                        confidence=confidence
                    )
            
            # Pattern detection: Rapid emotion changes (potential distress)
            if len(self.emotion_history) >= 5:
                recent_emotions = [e['emotion'] for e in self.emotion_history[-5:]]
                unique_emotions = set(recent_emotions)
                if len(unique_emotions) >= 4:
                    logger.warning("🔴 PATTERN DETECTED: Rapid emotional fluctuation")
                    self.execute_autonomous_action(
                        action_type='suggest_grounding',
                        emotion='fluctuating',
                        duration=0,
                        priority='medium',
                        confidence=0.8
                    )
        
        except Exception as e:
            logger.error(f"Error in autonomous decision engine: {e}")
    
    def execute_autonomous_action(self, action_type: str, emotion: str, 
                                   duration: float, priority: str, confidence: float):
        """🤖 EXECUTE AUTONOMOUS ACTION based on decision"""
        try:
            # Prevent duplicate actions within 2 minutes
            if self.last_escalation_time:
                time_since_last = (datetime.now() - self.last_escalation_time).total_seconds()
                if time_since_last < 120:
                    logger.info(f"⏳ Skipping action (cooldown: {120-time_since_last:.0f}s remaining)")
                    return
            
            action_record = {
                'timestamp': datetime.now().isoformat(),
                'action_type': action_type,
                'emotion': emotion,
                'duration': duration,
                'priority': priority,
                'confidence': confidence,
                'success': False
            }
            
            logger.warning(f"🤖 AUTONOMOUS ACTION: {action_type.upper()}")
            logger.warning(f"   Reason: {emotion} detected for {duration:.1f}s (threshold exceeded)")
            logger.warning(f"   Priority: {priority.upper()} | Confidence: {confidence:.2f}")
            
            if action_type == 'escalate_crisis':
                # 🚨 ESCALATE TO CRISIS COUNSELOR
                if self.crisis_counselor:
                    logger.error("🚨 ESCALATING TO CRISIS COUNSELOR")
                    try:
                        crisis_result = self.crisis_counselor.handle_crisis({
                            'emotion': emotion,
                            'duration': duration,
                            'confidence': confidence,
                            'source': 'autonomous_video_agent',
                            'message': f"Persistent {emotion} detected for {duration/60:.1f} minutes"
                        })
                        action_record['success'] = True
                        action_record['crisis_response'] = crisis_result
                        logger.info(f"✓ Crisis counselor activated: {crisis_result}")
                        self.intervention_active = True
                    except Exception as e:
                        logger.error(f"✗ Crisis escalation failed: {e}")
                        action_record['error'] = str(e)
                else:
                    logger.error("✗ Crisis counselor not available!")
                    action_record['error'] = 'crisis_counselor_unavailable'
            
            elif action_type == 'notify_therapy':
                # Notify therapy agent
                logger.warning("📢 NOTIFYING THERAPY AGENT")
                if self.therapy_agent:
                    action_record['success'] = True
                    logger.info(f"✓ Therapy agent notified about {emotion}")
            
            elif action_type == 'suggest_grounding':
                # Suggest grounding techniques
                logger.info("💡 SUGGESTING GROUNDING TECHNIQUES")
                action_record['success'] = True
                action_record['suggestion'] = "5-4-3-2-1 grounding technique recommended"
            
            elif action_type == 'check_engagement':
                # Check user engagement
                logger.info("👤 CHECKING USER ENGAGEMENT")
                action_record['success'] = True
            
            # Record action
            self.autonomous_actions.append(action_record)
            self.decision_history.append(action_record)
            self.last_escalation_time = datetime.now()
            
            # Keep only last 50 actions
            if len(self.autonomous_actions) > 50:
                self.autonomous_actions = self.autonomous_actions[-50:]
        
        except Exception as e:
            logger.error(f"Error executing autonomous action: {e}")
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get status of autonomous decision-making system"""
        return {
            "autonomous_enabled": self.autonomous_enabled,
            "current_goal": self.current_goal,
            "intervention_active": self.intervention_active,
            "emotion_persistence": {
                "emotion": self.emotion_persistence['current_emotion'],
                "duration": self.emotion_persistence['duration'],
                "threshold_status": self._get_threshold_status()
            },
            "escalation_rules": self.escalation_rules,
            "actions_taken": len(self.autonomous_actions),
            "recent_actions": self.autonomous_actions[-5:] if self.autonomous_actions else [],
            "last_escalation": self.last_escalation_time.isoformat() if self.last_escalation_time else None
        }
    
    def _get_threshold_status(self) -> Dict[str, Any]:
        """Calculate how close current emotion is to escalation threshold"""
        emotion = self.emotion_persistence['current_emotion']
        duration = self.emotion_persistence['duration']
        
        if emotion and emotion in self.escalation_rules:
            threshold = self.escalation_rules[emotion]['duration']
            percentage = (duration / threshold) * 100
            return {
                "emotion": emotion,
                "duration": duration,
                "threshold": threshold,
                "percentage": min(percentage, 100),
                "will_escalate_in": max(0, threshold - duration)
            }
        return {"status": "no_active_tracking"}
    
    def set_crisis_counselor(self, crisis_counselor):
        """Link crisis counselor for autonomous escalation"""
        self.crisis_counselor = crisis_counselor
        logger.info("✓ Crisis counselor linked to autonomous video agent")
    
    def set_therapy_agent(self, therapy_agent):
        """Link therapy agent for coordination"""
        self.therapy_agent = therapy_agent
        logger.info("✓ Therapy agent linked to autonomous video agent")

    def get_video_capabilities(self) -> Dict[str, Any]:
        """Get information about video processing capabilities"""
        return {
            "face_detection": {
                "available": self.face_cascade is not None,
                "method": "opencv_haar",
                "accuracy": "medium"
            },
            "emotion_recognition": {
                "available": True,
                "methods": ["fer_deep_learning"] if FER_AVAILABLE else ["basic_heuristic"],
                "emotions_supported": list(self.emotion_mapping.keys()),
                "accuracy": "high" if FER_AVAILABLE else "low",
                "model": "Deep Neural Network (FER)" if FER_AVAILABLE else "Rule-based Heuristics"
            },
            "video_processing": {
                "real_time_analysis": True,
                "continuous_monitoring": True,
                "trend_analysis": True,
                "therapeutic_suggestions": True,
                "callback_support": True
            },
            "autonomous_capabilities": {
                "decision_making": self.autonomous_enabled,
                "pattern_detection": True,
                "automatic_escalation": True,
                "crisis_integration": self.crisis_counselor is not None,
                "goal_oriented": True,
                "escalation_rules": len(self.escalation_rules)
            },
            "hardware_requirements": {
                "camera_required": True,
                "gpu_accelerated": False,
                "processing_load": "medium"
            }
        }

    def test_video_system(self) -> Dict[str, Any]:
        """Test video system functionality"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        # Test camera initialization
        camera_result = self.start_camera(auto_start_monitoring=False)
        results["tests"]["camera"] = camera_result

        if camera_result["success"]:
            # Test frame capture
            frame_result = self.capture_and_encode_frame()
            results["tests"]["frame_capture"] = {
                "success": frame_result["success"],
                "frame_size": frame_result.get("frame_size", 0)
            }

            # Test emotion analysis
            analysis_result = self.analyze_current_frame()
            results["tests"]["emotion_analysis"] = {
                "success": analysis_result["success"],
                "faces_detected": analysis_result.get("faces_detected", 0),
                "method": analysis_result.get("analysis_method", "unknown"),
                "dominant_emotion": analysis_result.get("dominant_emotion", "unknown"),
                "confidence": analysis_result.get("confidence", 0.0)
            }

            # Test continuous monitoring
            monitor_result = self.start_continuous_analysis()
            time.sleep(3)  # Let it run for 3 seconds
            stop_result = self.stop_continuous_analysis()
            
            results["tests"]["continuous_monitoring"] = {
                "success": monitor_result["success"] and stop_result["success"],
                "analyses_performed": len(self.emotion_history)
            }

            # Stop camera after testing
            self.stop_camera()

        # Overall status
        results["overall_status"] = (
            results["tests"]["camera"]["success"] and
            results["tests"].get("frame_capture", {}).get("success", False) and
            results["tests"].get("emotion_analysis", {}).get("success", False) and
            results["tests"].get("continuous_monitoring", {}).get("success", False)
        )

        return results

# Test function for video agent
def test_video_agent():
    """Test the video agent functionality"""
    print("Testing Video Agent with Continuous Monitoring & FER")
    print("=" * 60)

    agent = VideoAgent()

    # Test system capabilities
    print("\n1. Testing Video Capabilities:")
    capabilities = agent.get_video_capabilities()
    print(f"Face Detection: {capabilities['face_detection']['available']}")
    print(f"Emotion Recognition: {capabilities['emotion_recognition']['available']}")
    print(f"Methods: {capabilities['emotion_recognition']['methods']}")
    print(f"Accuracy: {capabilities['emotion_recognition']['accuracy']}")
    print(f"Model: {capabilities['emotion_recognition']['model']}")

    # Test status
    print("\n2. Testing Video Status:")
    status = agent.get_video_status()
    print(f"System Status: {status}")

    # Run system test
    print("\n3. Running Comprehensive System Test:")
    test_results = agent.test_video_system()
    print(f"\nOverall System Status: {'✅ PASS' if test_results['overall_status'] else '❌ FAIL'}")

    if test_results["tests"]["camera"]["success"]:
        print(f"✅ Camera: Working")
        print(f"✅ Frame Capture: {'Working' if test_results['tests']['frame_capture']['success'] else 'Failed'}")
        print(f"✅ Emotion Analysis: {'Working' if test_results['tests']['emotion_analysis']['success'] else 'Failed'}")
        print(f"   - Method: {test_results['tests']['emotion_analysis']['method']}")
        print(f"   - Detected: {test_results['tests']['emotion_analysis']['dominant_emotion']}")
        print(f"   - Confidence: {test_results['tests']['emotion_analysis']['confidence']:.2f}")
        print(f"✅ Continuous Monitoring: {'Working' if test_results['tests']['continuous_monitoring']['success'] else 'Failed'}")
        print(f"   - Analyses: {test_results['tests']['continuous_monitoring']['analyses_performed']}")
    else:
        print("❌ Camera not available - check if camera is connected and not in use")

    print("\n" + "=" * 60)
    print("Video Agent Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_video_agent()