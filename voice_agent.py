"""
Voice Agent for Agentic Therapy AI System
Handles speech-to-text and text-to-speech functionality for voice interactions
"""

import logging
import speech_recognition as sr
import pyttsx3
import threading
import queue
import os
import tempfile
import wave
import json
from typing import Dict, Optional, Any
import time
from datetime import datetime
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAgent:
    """
    Voice processing agent for therapeutic AI system
    Handles speech recognition and text-to-speech synthesis
    """

    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False
        self._tts_lock = threading.Lock()
        self._tts_queue = queue.Queue()
        self._tts_worker_thread = None
        self._stop_tts_worker = False

        # Voice settings
        self.voice_settings = self.config.get("voice", {
            "recognition_timeout": 5,
            "recognition_phrase_timeout": 1,
            "tts_rate": 180,
            "tts_volume": 0.8,
            "tts_voice_gender": "female",
            "noise_adjustment_duration": 1,
            "language": "en-US"
        })

        # Initialize components
        self.init_speech_recognition()
        self.init_text_to_speech()
        self.init_tts_worker()

        logger.info("Voice agent initialized successfully")

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

    def init_speech_recognition(self):
        """Initialize speech recognition components"""
        try:
            # Initialize microphone
            self.microphone = sr.Microphone()

            # Adjust for ambient noise
            logger.info("Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=self.voice_settings["noise_adjustment_duration"]
                )

            # Configure recognizer settings
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3

            logger.info("✓ Speech recognition initialized")

        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
            self.microphone = None

    def init_text_to_speech(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()

            # Configure TTS settings
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', self.voice_settings["tts_rate"])

            volume = self.tts_engine.getProperty('volume')
            self.tts_engine.setProperty('volume', self.voice_settings["tts_volume"])

            # Set voice (prefer female voice for therapy)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Fallback to first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)

            logger.info("✓ Text-to-speech initialized")

        except Exception as e:
            logger.error(f"Failed to initialize text-to-speech: {e}")
            self.tts_engine = None

    def init_tts_worker(self):
        """Initialize TTS worker thread to handle speech synthesis queue"""
        try:
            def tts_worker():
                """Worker thread function to process TTS queue"""
                import platform
                thread_engine = None
                
                # Windows requires COM initialization in each thread
                if platform.system() == 'Windows':
                    try:
                        import pythoncom
                        pythoncom.CoInitialize()
                        logger.info("COM initialized for TTS worker thread")
                    except ImportError:
                        logger.warning("pythoncom not available - TTS may not work correctly")
                    except Exception as e:
                        logger.warning(f"COM initialization warning: {e}")
                
                while not self._stop_tts_worker:
                    try:
                        # Wait for TTS requests (timeout to check stop condition)
                        text = self._tts_queue.get(timeout=1.0)

                        if text:
                            try:
                                self.is_speaking = True
                                logger.info(f"TTS worker processing: '{text[:30]}...'")
                                
                                # Create engine in worker thread for Windows
                                try:
                                    if thread_engine is None:
                                        thread_engine = pyttsx3.init()
                                        thread_engine.setProperty('rate', self.voice_settings["tts_rate"])
                                        thread_engine.setProperty('volume', self.voice_settings["tts_volume"])
                                        # Set voice
                                        voices = thread_engine.getProperty('voices')
                                        if voices:
                                            for voice in voices:
                                                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                                    thread_engine.setProperty('voice', voice.id)
                                                    break
                                            else:
                                                thread_engine.setProperty('voice', voices[0].id)
                                    
                                    thread_engine.say(text)
                                    thread_engine.runAndWait()
                                    logger.info(f"TTS completed for text: '{text[:30]}...'")
                                except Exception as e:
                                    logger.error(f"TTS engine error: {e}")
                                    # Reset engine on error
                                    thread_engine = None
                                        
                            except Exception as e:
                                logger.error(f"Error in TTS worker: {e}")
                            finally:
                                self.is_speaking = False
                                self._tts_queue.task_done()

                    except queue.Empty:
                        # Timeout - continue loop to check stop condition
                        continue
                    except Exception as e:
                        logger.error(f"Error in TTS worker thread: {e}")
                
                # Cleanup
                if thread_engine:
                    try:
                        thread_engine.stop()
                    except:
                        pass
                
                # Windows COM cleanup
                if platform.system() == 'Windows':
                    try:
                        import pythoncom
                        pythoncom.CoUninitialize()
                    except:
                        pass

            # Start the worker thread
            self._tts_worker_thread = threading.Thread(target=tts_worker, daemon=True)
            self._tts_worker_thread.start()
            logger.info("✓ TTS worker thread initialized")

        except Exception as e:
            logger.error(f"Failed to initialize TTS worker: {e}")

    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice system status"""
        return {
            "speech_recognition_available": self.microphone is not None,
            "text_to_speech_available": self.tts_engine is not None,
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking,
            "settings": self.voice_settings
        }

    def start_listening(self) -> Dict[str, Any]:
        """Start listening for voice input"""
        if not self.microphone:
            return {
                "success": False,
                "error": "Microphone not available"
            }

        if self.is_listening:
            return {
                "success": False,
                "error": "Already listening"
            }

        try:
            self.is_listening = True
            logger.info("Started listening for voice input...")

            return {
                "success": True,
                "message": "Listening started",
                "status": "listening"
            }

        except Exception as e:
            logger.error(f"Error starting to listen: {e}")
            self.is_listening = False
            return {
                "success": False,
                "error": f"Failed to start listening: {str(e)}"
            }

    def stop_listening(self) -> Dict[str, Any]:
        """Stop listening for voice input"""
        self.is_listening = False
        logger.info("Stopped listening for voice input")

        return {
            "success": True,
            "message": "Listening stopped",
            "status": "idle"
        }

    def recognize_speech_from_audio(self, audio_data) -> Dict[str, Any]:
        """Convert audio data to text using speech recognition"""
        if not self.microphone:
            return {
                "success": False,
                "error": "Speech recognition not available"
            }

        try:
            logger.info("Processing audio for speech recognition...")
            start_time = time.time()

            # Use Google Speech Recognition (free)
            try:
                text = self.recognizer.recognize_google(
                    audio_data,
                    language=self.voice_settings["language"]
                )
                processing_time = time.time() - start_time

                logger.info(f"Speech recognized in {processing_time:.2f}s: '{text[:50]}...'")

                return {
                    "success": True,
                    "text": text,
                    "confidence": 0.8,  # Google API doesn't return confidence
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }

            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return {
                    "success": False,
                    "error": "Could not understand speech",
                    "error_type": "unclear_speech"
                }

            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                # Fallback to offline recognition if available
                try:
                    text = self.recognizer.recognize_sphinx(audio_data)
                    return {
                        "success": True,
                        "text": text,
                        "confidence": 0.6,
                        "provider": "offline",
                        "timestamp": datetime.now().isoformat()
                    }
                except:
                    return {
                        "success": False,
                        "error": f"Speech recognition failed: {str(e)}",
                        "error_type": "service_error"
                    }

        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return {
                "success": False,
                "error": f"Recognition error: {str(e)}",
                "error_type": "processing_error"
            }

    def capture_audio(self, duration: float = None) -> Dict[str, Any]:
        """Capture audio from microphone"""
        if not self.microphone:
            return {
                "success": False,
                "error": "Microphone not available"
            }

        try:
            logger.info("Listening for audio input...")

            with self.microphone as source:
                if duration:
                    # Record for specific duration
                    audio = self.recognizer.record(source, duration=duration)
                else:
                    # Listen for speech with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.voice_settings["recognition_timeout"],
                        phrase_time_limit=self.voice_settings["recognition_phrase_timeout"]
                    )

            # Process the captured audio
            return self.recognize_speech_from_audio(audio)

        except sr.WaitTimeoutError:
            logger.warning("Listening timeout - no speech detected")
            return {
                "success": False,
                "error": "No speech detected within timeout",
                "error_type": "timeout"
            }

        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
            return {
                "success": False,
                "error": f"Audio capture failed: {str(e)}",
                "error_type": "capture_error"
            }

    def speak_text(self, text: str, async_mode: bool = True) -> Dict[str, Any]:
        """Convert text to speech and play it"""
        if not text or not text.strip():
            return {
                "success": False,
                "error": "No text provided"
            }

        try:
            logger.info(f"Speaking text: '{text[:50]}...'")
            
            import platform
            
            # On Windows, use synchronous mode to avoid threading issues with pyttsx3
            if platform.system() == 'Windows':
                try:
                    self.is_speaking = True
                    # Create a fresh engine for each call on Windows
                    engine = pyttsx3.init()
                    engine.setProperty('rate', self.voice_settings["tts_rate"])
                    engine.setProperty('volume', self.voice_settings["tts_volume"])
                    
                    # Set voice
                    voices = engine.getProperty('voices')
                    if voices:
                        for voice in voices:
                            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                engine.setProperty('voice', voice.id)
                                break
                        else:
                            engine.setProperty('voice', voices[0].id)
                    
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                    
                    logger.info(f"TTS completed for text: '{text[:30]}...'")
                    
                    return {
                        "success": True,
                        "message": "Speech synthesis completed",
                        "text_length": len(text),
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Windows TTS error: {e}")
                    return {
                        "success": False,
                        "error": f"TTS error: {str(e)}"
                    }
                finally:
                    self.is_speaking = False
            else:
                # Non-Windows: use queue-based async mode
                self._tts_queue.put(text)
                return {
                    "success": True,
                    "message": "Speech synthesis queued",
                    "text_length": len(text),
                    "async": True,
                    "timestamp": datetime.now().isoformat(),
                    "queue_size": self._tts_queue.qsize()
                }

        except Exception as e:
            logger.error(f"Error in speak_text: {e}")
            return {
                "success": False,
                "error": f"TTS error: {str(e)}"
            }

    def stop_speaking(self) -> Dict[str, Any]:
        """Stop current speech synthesis and clear queue"""
        try:
            # Clear the TTS queue
            while not self._tts_queue.empty():
                try:
                    self._tts_queue.get_nowait()
                    self._tts_queue.task_done()
                except queue.Empty:
                    break

            if self.tts_engine and self.is_speaking:
                self.tts_engine.stop()
                self.is_speaking = False

                return {
                    "success": True,
                    "message": "Speech stopped and queue cleared"
                }
            else:
                return {
                    "success": True,
                    "message": "Speech queue cleared"
                }

        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
            return {
                "success": False,
                "error": f"Failed to stop speech: {str(e)}"
            }

    def cleanup(self):
        """Cleanup voice agent resources"""
        try:
            self._stop_tts_worker = True
            if self._tts_worker_thread and self._tts_worker_thread.is_alive():
                self._tts_worker_thread.join(timeout=2.0)

            if self.tts_engine:
                self.tts_engine.stop()

            logger.info("Voice agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def process_voice_interaction(self) -> Dict[str, Any]:
        """Complete voice interaction cycle: listen -> recognize -> return result"""
        try:
            # Start listening
            listen_result = self.start_listening()
            if not listen_result["success"]:
                return listen_result

            # Capture and recognize audio
            recognition_result = self.capture_audio()

            # Stop listening
            self.stop_listening()

            return recognition_result

        except Exception as e:
            logger.error(f"Error in voice interaction: {e}")
            self.stop_listening()
            return {
                "success": False,
                "error": f"Voice interaction failed: {str(e)}"
            }

    def get_voice_capabilities(self) -> Dict[str, Any]:
        """Get information about voice capabilities"""
        capabilities = {
            "speech_recognition": {
                "available": self.microphone is not None,
                "providers": ["google", "offline_sphinx"],
                "languages": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE"],
                "current_language": self.voice_settings["language"]
            },
            "text_to_speech": {
                "available": self.tts_engine is not None,
                "voices_count": 0,
                "current_settings": {
                    "rate": self.voice_settings["tts_rate"],
                    "volume": self.voice_settings["tts_volume"]
                }
            },
            "real_time_features": {
                "continuous_listening": True,
                "interrupt_capability": True,
                "async_speech": True
            }
        }

        # Get available voices
        if self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                capabilities["text_to_speech"]["voices_count"] = len(voices) if voices else 0
                capabilities["text_to_speech"]["available_voices"] = [
                    {
                        "id": voice.id,
                        "name": voice.name,
                        "gender": "female" if "female" in voice.name.lower() else "male"
                    } for voice in voices[:5]  # Limit to first 5
                ] if voices else []
            except Exception as e:
                logger.warning(f"Could not get voice information: {e}")

        return capabilities

    def test_voice_system(self) -> Dict[str, Any]:
        """Test voice system functionality"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        # Test speech recognition
        if self.microphone:
            results["tests"]["speech_recognition"] = {
                "available": True,
                "microphone_detected": True
            }
        else:
            results["tests"]["speech_recognition"] = {
                "available": False,
                "error": "Microphone not available"
            }

        # Test text-to-speech
        if self.tts_engine:
            try:
                test_result = self.speak_text("Voice system test successful", async_mode=False)
                results["tests"]["text_to_speech"] = {
                    "available": True,
                    "test_speech": test_result["success"]
                }
            except Exception as e:
                results["tests"]["text_to_speech"] = {
                    "available": True,
                    "test_speech": False,
                    "error": str(e)
                }
        else:
            results["tests"]["text_to_speech"] = {
                "available": False,
                "error": "TTS engine not available"
            }

        # Overall status
        results["overall_status"] = (
            results["tests"]["speech_recognition"]["available"] and
            results["tests"]["text_to_speech"]["available"]
        )

        return results

# Test function for voice agent
def test_voice_agent():
    """Test the voice agent functionality"""
    print("Testing Voice Agent")
    print("=" * 50)

    agent = VoiceAgent()

    # Test system capabilities
    print("\n1. Testing Voice Capabilities:")
    capabilities = agent.get_voice_capabilities()
    print(f"Speech Recognition: {capabilities['speech_recognition']['available']}")
    print(f"Text-to-Speech: {capabilities['text_to_speech']['available']}")
    print(f"Available Voices: {capabilities['text_to_speech']['voices_count']}")

    # Test status
    print("\n2. Testing Voice Status:")
    status = agent.get_voice_status()
    print(f"System Status: {status}")

    # Test TTS
    print("\n3. Testing Text-to-Speech:")
    tts_result = agent.speak_text("Hello! This is a test of the voice system for therapeutic AI.", async_mode=False)
    print(f"TTS Result: {tts_result}")

    # Test system
    print("\n4. Running System Test:")
    test_results = agent.test_voice_system()
    print(f"Overall System Status: {'✓ PASS' if test_results['overall_status'] else '❌ FAIL'}")

    print("\nVoice Agent Test Complete!")

if __name__ == "__main__":
    test_voice_agent()