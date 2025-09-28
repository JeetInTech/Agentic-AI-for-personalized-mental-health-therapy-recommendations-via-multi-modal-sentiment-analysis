"""
Video Agent for Agentic Therapy AI System
Handles facial expression recognition and emotional analysis from video streams
"""

import logging
import cv2
import numpy as np
import threading
import queue
import base64
import json
import time
from typing import Dict, List, Optional, Any, Tuple
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
except ImportError:
    FER_AVAILABLE = False
    logging.warning("fer not available - using OpenCV-based emotion detection")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAgent:
    """
    Video processing agent for therapeutic AI system
    Handles facial expression recognition and emotion analysis
    """

    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.camera = None
        self.is_recording = False
        self.is_analyzing = False

        # Video settings
        self.video_settings = self.config.get("video", {
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30,
            "analysis_interval": 1.0,  # Analyze every 1 second
            "emotion_threshold": 0.4,
            "face_detection_scale": 1.1,
            "min_neighbors": 5
        })

        # Emotion mapping
        self.emotion_mapping = {
            'angry': {'valence': -0.8, 'arousal': 0.8, 'therapy_priority': 'high'},
            'disgust': {'valence': -0.6, 'arousal': 0.4, 'therapy_priority': 'medium'},
            'fear': {'valence': -0.7, 'arousal': 0.9, 'therapy_priority': 'high'},
            'happy': {'valence': 0.8, 'arousal': 0.6, 'therapy_priority': 'low'},
            'sad': {'valence': -0.8, 'arousal': -0.4, 'therapy_priority': 'high'},
            'surprise': {'valence': 0.2, 'arousal': 0.8, 'therapy_priority': 'low'},
            'neutral': {'valence': 0.0, 'arousal': 0.0, 'therapy_priority': 'low'}
        }

        # Analysis history
        self.emotion_history = []
        self.analysis_queue = queue.Queue(maxsize=100)

        # Initialize components
        self.face_cascade = None
        self.emotion_detector = None

        self.init_face_detection()
        self.init_emotion_detection()

        logger.info("Video agent initialized successfully")

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
        """Initialize emotion detection system"""
        try:
            if FER_AVAILABLE:
                # Use FER library for emotion detection
                self.emotion_detector = FER(mtcnn=False)  # Use OpenCV for face detection
                logger.info("✓ Emotion detection initialized (FER library)")
            else:
                # Fallback to rule-based emotion detection
                self.emotion_detector = None
                logger.info("✓ Using rule-based emotion analysis (FER not available)")

        except Exception as e:
            logger.error(f"Failed to initialize emotion detection: {e}")
            self.emotion_detector = None

    def get_video_status(self) -> Dict[str, Any]:
        """Get current video system status"""
        return {
            "camera_available": self.camera is not None and self.camera.isOpened() if self.camera else False,
            "face_detection_available": self.face_cascade is not None,
            "emotion_detection_available": self.emotion_detector is not None or self.face_cascade is not None,
            "is_recording": self.is_recording,
            "is_analyzing": self.is_analyzing,
            "settings": self.video_settings,
            "libraries_status": {
                "opencv": True,  # Required for this agent
                "dlib": DLIB_AVAILABLE,
                "fer": FER_AVAILABLE
            }
        }

    def start_camera(self) -> Dict[str, Any]:
        """Initialize and start camera"""
        try:
            if self.camera and self.camera.isOpened():
                return {
                    "success": True,
                    "message": "Camera already running"
                }

            # Initialize camera
            camera_index = self.video_settings["camera_index"]
            self.camera = cv2.VideoCapture(camera_index)

            if not self.camera.isOpened():
                return {
                    "success": False,
                    "error": f"Failed to open camera at index {camera_index}"
                }

            # Configure camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_settings["frame_width"])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_settings["frame_height"])
            self.camera.set(cv2.CAP_PROP_FPS, self.video_settings["fps"])

            # Test camera by taking a frame
            ret, frame = self.camera.read()
            if not ret:
                self.camera.release()
                self.camera = None
                return {
                    "success": False,
                    "error": "Camera opened but cannot read frames"
                }

            logger.info(f"Camera initialized: {frame.shape[1]}x{frame.shape[0]}")

            return {
                "success": True,
                "message": "Camera started successfully",
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "camera_index": camera_index
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
            if self.camera and self.camera.isOpened():
                self.camera.release()
                self.camera = None
                self.is_recording = False
                self.is_analyzing = False
                logger.info("Camera stopped and resources released")

            return {
                "success": True,
                "message": "Camera stopped"
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
        """Analyze emotions using FER library"""
        if not FER_AVAILABLE or self.emotion_detector is None:
            return self.analyze_emotion_basic(frame)

        try:
            # Analyze emotions using FER
            emotions = self.emotion_detector.detect_emotions(frame)

            if not emotions:
                return {
                    "faces_detected": 0,
                    "emotions": [],
                    "dominant_emotion": "neutral",
                    "confidence": 0.0,
                    "analysis_method": "fer"
                }

            # Process each face
            face_emotions = []
            for face_data in emotions:
                emotions_dict = face_data['emotions']

                # Find dominant emotion
                dominant = max(emotions_dict, key=emotions_dict.get)
                confidence = emotions_dict[dominant]

                # Get face box
                box = face_data['box']

                face_emotions.append({
                    "face_box": box,
                    "emotions": emotions_dict,
                    "dominant_emotion": dominant,
                    "confidence": confidence,
                    "therapy_analysis": self.get_therapy_analysis(dominant, confidence)
                })

            # Overall analysis
            if face_emotions:
                # Use first face for overall emotion
                primary_face = face_emotions[0]
                return {
                    "faces_detected": len(face_emotions),
                    "emotions": face_emotions,
                    "dominant_emotion": primary_face["dominant_emotion"],
                    "confidence": primary_face["confidence"],
                    "analysis_method": "fer",
                    "therapy_priority": primary_face["therapy_analysis"]["priority"],
                    "therapeutic_suggestion": primary_face["therapy_analysis"]["suggestion"]
                }
            else:
                return {
                    "faces_detected": 0,
                    "emotions": [],
                    "dominant_emotion": "neutral",
                    "confidence": 0.0,
                    "analysis_method": "fer"
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
                    "analysis_method": "basic"
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
                    emotion = "surprised"
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
            return {
                "faces_detected": len(face_emotions),
                "emotions": face_emotions,
                "dominant_emotion": primary_face["dominant_emotion"],
                "confidence": primary_face["confidence"],
                "analysis_method": "basic_heuristic",
                "therapy_priority": primary_face["therapy_analysis"]["priority"],
                "therapeutic_suggestion": primary_face["therapy_analysis"]["suggestion"]
            }

        except Exception as e:
            logger.error(f"Error in basic emotion analysis: {e}")
            return {
                "faces_detected": 0,
                "emotions": [],
                "dominant_emotion": "neutral",
                "confidence": 0.0,
                "analysis_method": "error",
                "error": str(e)
            }

    def get_therapy_analysis(self, emotion: str, confidence: float) -> Dict[str, Any]:
        """Get therapeutic analysis for detected emotion"""
        emotion_info = self.emotion_mapping.get(emotion, self.emotion_mapping['neutral'])

        # Determine intervention suggestions
        suggestions = {
            'angry': "Consider deep breathing exercises or progressive muscle relaxation",
            'sad': "Would you like to talk about what's making you feel this way?",
            'fear': "Let's work on some grounding techniques to help you feel more secure",
            'happy': "I'm glad to see you're feeling positive! What's contributing to your good mood?",
            'surprise': "You seem surprised - is there something unexpected we should discuss?",
            'neutral': "How are you feeling today? I'm here to listen",
            'disgust': "Something seems to be bothering you - would you like to explore that?"
        }

        return {
            "priority": emotion_info["therapy_priority"],
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "suggestion": suggestions.get(emotion, "Let's talk about how you're feeling"),
            "confidence_level": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
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

            # Add to history
            self.emotion_history.append({
                "timestamp": analysis["timestamp"],
                "emotion": analysis["dominant_emotion"],
                "confidence": analysis["confidence"]
            })

            # Keep only recent history (last 100 analyses)
            if len(self.emotion_history) > 100:
                self.emotion_history = self.emotion_history[-100:]

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

            return {
                "success": True,
                "duration_minutes": duration_minutes,
                "total_analyses": total_count,
                "trends": {
                    "dominant_emotion": dominant_emotion,
                    "average_confidence": avg_confidence,
                    "emotion_distribution": emotion_percentages,
                    "emotion_counts": emotion_counts,
                    "therapy_recommendations": self.get_trend_therapy_recommendations(emotion_percentages)
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
            recommendations.append("Frequent sadness detected - consider exploring underlying causes")

        if emotion_percentages.get('angry', 0) > 30:
            recommendations.append("Anger patterns noticed - anger management techniques might be helpful")

        if emotion_percentages.get('fear', 0) > 25:
            recommendations.append("Anxiety/fear levels elevated - relaxation techniques recommended")

        if emotion_percentages.get('neutral', 0) > 70:
            recommendations.append("Emotional expression seems limited - consider exploring feelings more deeply")

        if emotion_percentages.get('happy', 0) > 60:
            recommendations.append("Positive emotional state - good time to reinforce coping strategies")

        if not recommendations:
            recommendations.append("Emotional patterns appear balanced")

        return recommendations

    def start_continuous_analysis(self, callback=None) -> Dict[str, Any]:
        """Start continuous emotion analysis in background thread"""
        if self.is_analyzing:
            return {
                "success": False,
                "error": "Analysis already running"
            }

        if not self.camera or not self.camera.isOpened():
            camera_result = self.start_camera()
            if not camera_result["success"]:
                return camera_result

        def analysis_thread():
            self.is_analyzing = True
            last_analysis = 0

            try:
                while self.is_analyzing:
                    current_time = time.time()

                    # Analyze at specified interval
                    if current_time - last_analysis >= self.video_settings["analysis_interval"]:
                        result = self.analyze_current_frame()

                        if result["success"] and callback:
                            callback(result)

                        last_analysis = current_time

                    time.sleep(0.1)  # Short sleep to prevent excessive CPU usage

            except Exception as e:
                logger.error(f"Error in analysis thread: {e}")
            finally:
                self.is_analyzing = False

        # Start analysis thread
        thread = threading.Thread(target=analysis_thread, daemon=True)
        thread.start()

        return {
            "success": True,
            "message": "Continuous analysis started",
            "analysis_interval": self.video_settings["analysis_interval"]
        }

    def stop_continuous_analysis(self) -> Dict[str, Any]:
        """Stop continuous emotion analysis"""
        self.is_analyzing = False

        return {
            "success": True,
            "message": "Continuous analysis stopped"
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
                "methods": ["fer_library"] if FER_AVAILABLE else ["basic_heuristic"],
                "emotions_supported": list(self.emotion_mapping.keys()),
                "accuracy": "high" if FER_AVAILABLE else "low"
            },
            "video_processing": {
                "real_time_analysis": True,
                "trend_analysis": True,
                "therapeutic_suggestions": True
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
        camera_result = self.start_camera()
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
                "method": analysis_result.get("analysis_method", "unknown")
            }

            # Stop camera after testing
            self.stop_camera()

        # Overall status
        results["overall_status"] = (
            results["tests"]["camera"]["success"] and
            results["tests"].get("frame_capture", {}).get("success", False) and
            results["tests"].get("emotion_analysis", {}).get("success", False)
        )

        return results

# Test function for video agent
def test_video_agent():
    """Test the video agent functionality"""
    print("Testing Video Agent")
    print("=" * 50)

    agent = VideoAgent()

    # Test system capabilities
    print("\n1. Testing Video Capabilities:")
    capabilities = agent.get_video_capabilities()
    print(f"Face Detection: {capabilities['face_detection']['available']}")
    print(f"Emotion Recognition: {capabilities['emotion_recognition']['available']}")
    print(f"Methods: {capabilities['emotion_recognition']['methods']}")

    # Test status
    print("\n2. Testing Video Status:")
    status = agent.get_video_status()
    print(f"System Status: {status}")

    # Run system test
    print("\n3. Running System Test:")
    test_results = agent.test_video_system()
    print(f"Overall System Status: {'✓ PASS' if test_results['overall_status'] else '❌ FAIL'}")

    if test_results["tests"]["camera"]["success"]:
        print(f"Camera: ✓ Working")
        print(f"Frame Capture: {'✓ Working' if test_results['tests']['frame_capture']['success'] else '❌ Failed'}")
        print(f"Emotion Analysis: {'✓ Working' if test_results['tests']['emotion_analysis']['success'] else '❌ Failed'}")
    else:
        print("❌ Camera not available - check if camera is connected and not in use by other applications")

    print("\nVideo Agent Test Complete!")

if __name__ == "__main__":
    test_video_agent()