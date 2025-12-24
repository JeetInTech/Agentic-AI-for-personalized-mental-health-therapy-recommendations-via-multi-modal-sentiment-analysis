"""
Real-time Conversational Voice Agent
Provides ChatGPT-like voice conversation with interrupt capability

Features:
- Real-time speech-to-text streaming
- Interruptible text-to-speech (stops when user speaks)
- Continuous listening mode
- WebSocket-based low-latency communication
"""

import logging
import threading
import queue
import json
import time
import os
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from enum import Enum

# Speech Recognition
import speech_recognition as sr

# Text-to-Speech options
import pyttsx3

# For async TTS with interrupt support
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available - using pyttsx3 for TTS")

try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logging.warning("gTTS not available")

try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.warning("edge-tts not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceState(Enum):
    """Voice system states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class RealtimeVoiceAgent:
    """
    Real-time voice conversation agent with interrupt capability
    Like ChatGPT's voice mode - talk, get response, interrupt anytime
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        
        # State management
        self.state = VoiceState.IDLE
        self._state_lock = threading.Lock()
        self._state_callbacks: list[Callable] = []
        
        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self._listening_thread = None
        self._stop_listening = threading.Event()
        
        # TTS
        self.tts_engine = None
        self._tts_thread = None
        self._stop_speaking = threading.Event()
        self._current_audio = None
        
        # Conversation
        self._on_user_speech: Optional[Callable] = None
        self._on_response_ready: Optional[Callable] = None
        
        # Voice settings
        self.voice_settings = self.config.get("voice", {})
        self.tts_rate = self.voice_settings.get("tts_rate", 180)
        self.tts_volume = self.voice_settings.get("tts_volume", 0.8)
        self.language = self.voice_settings.get("language", "en-US")
        
        # Audio queue for TTS
        self._audio_queue = queue.Queue()
        
        # Initialize components
        self._init_speech_recognition()
        self._init_tts()
        
        logger.info("✓ Realtime Voice Agent initialized")

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def _init_speech_recognition(self):
        """Initialize speech recognition with optimized settings"""
        try:
            self.microphone = sr.Microphone()
            
            # Calibrate for ambient noise
            logger.info("Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Optimize for real-time conversation
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5  # Shorter pause = faster response
            self.recognizer.phrase_threshold = 0.2
            self.recognizer.non_speaking_duration = 0.3
            
            logger.info("✓ Speech recognition ready")
        except Exception as e:
            logger.error(f"Failed to init speech recognition: {e}")
            self.microphone = None

    def _init_tts(self):
        """Initialize TTS with interrupt support"""
        try:
            # Try Edge TTS first (best quality, async)
            if EDGE_TTS_AVAILABLE and PYGAME_AVAILABLE:
                self.tts_provider = "edge"
                logger.info("✓ Using Edge TTS (high quality)")
            # Fallback to gTTS + pygame
            elif GTTS_AVAILABLE and PYGAME_AVAILABLE:
                self.tts_provider = "gtts"
                logger.info("✓ Using gTTS")
            # Final fallback to pyttsx3
            else:
                self.tts_provider = "pyttsx3"
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.tts_rate)
                self.tts_engine.setProperty('volume', self.tts_volume)
                logger.info("✓ Using pyttsx3 TTS")
        except Exception as e:
            logger.error(f"TTS initialization error: {e}")
            self.tts_provider = None

    def _set_state(self, new_state: VoiceState):
        """Thread-safe state update with callbacks"""
        with self._state_lock:
            old_state = self.state
            self.state = new_state
            logger.info(f"Voice state: {old_state.value} -> {new_state.value}")
        
        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                callback(new_state.value)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def get_state(self) -> str:
        """Get current voice state"""
        with self._state_lock:
            return self.state.value

    def on_state_change(self, callback: Callable):
        """Register state change callback"""
        self._state_callbacks.append(callback)

    def set_speech_callback(self, callback: Callable):
        """Set callback for when user speech is recognized"""
        self._on_user_speech = callback

    def start_conversation(self) -> Dict[str, Any]:
        """Start voice conversation mode - continuous listening"""
        if self.microphone is None:
            return {"success": False, "error": "Microphone not available"}
        
        if self.state != VoiceState.IDLE:
            return {"success": False, "error": f"Cannot start, current state: {self.state.value}"}
        
        self._stop_listening.clear()
        self._listening_thread = threading.Thread(target=self._continuous_listen, daemon=True)
        self._listening_thread.start()
        
        self._set_state(VoiceState.LISTENING)
        
        return {
            "success": True,
            "message": "Voice conversation started",
            "state": self.state.value
        }

    def stop_conversation(self) -> Dict[str, Any]:
        """Stop voice conversation mode"""
        self._stop_listening.set()
        self._stop_speaking.set()
        
        # Stop any current audio
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        self._set_state(VoiceState.IDLE)
        
        return {
            "success": True,
            "message": "Voice conversation stopped",
            "state": self.state.value
        }

    def interrupt(self) -> Dict[str, Any]:
        """Interrupt current speech and start listening"""
        logger.info("🛑 Interrupt triggered!")
        
        self._stop_speaking.set()
        
        # Stop audio immediately
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        self._set_state(VoiceState.INTERRUPTED)
        
        # Clear speaking flag and resume listening
        time.sleep(0.1)  # Brief pause
        self._stop_speaking.clear()
        
        if not self._stop_listening.is_set():
            self._set_state(VoiceState.LISTENING)
        
        return {
            "success": True,
            "message": "Speech interrupted",
            "state": self.state.value
        }

    def _continuous_listen(self):
        """Continuous listening loop - runs in background thread"""
        logger.info("🎤 Starting continuous listening...")
        
        while not self._stop_listening.is_set():
            # Skip if currently speaking (unless interrupted)
            if self.state == VoiceState.SPEAKING:
                time.sleep(0.1)
                continue
            
            try:
                self._set_state(VoiceState.LISTENING)
                
                with self.microphone as source:
                    logger.debug("Listening for speech...")
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=5,
                            phrase_time_limit=15  # Max 15 seconds per phrase
                        )
                    except sr.WaitTimeoutError:
                        continue  # No speech detected, keep listening
                
                if self._stop_listening.is_set():
                    break
                
                # If we started speaking during listen, skip processing
                if self.state == VoiceState.SPEAKING:
                    continue
                
                self._set_state(VoiceState.PROCESSING)
                
                # Recognize speech
                try:
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    logger.info(f"🗣️ User said: {text}")
                    
                    # Check for interrupt while speaking
                    if self.state == VoiceState.SPEAKING:
                        self.interrupt()
                    
                    # Callback with recognized text
                    if self._on_user_speech and text.strip():
                        self._on_user_speech(text)
                    
                except sr.UnknownValueError:
                    logger.debug("Could not understand audio")
                except sr.RequestError as e:
                    logger.error(f"Speech recognition error: {e}")
                
            except Exception as e:
                logger.error(f"Listening error: {e}")
                time.sleep(0.5)
        
        logger.info("🎤 Stopped continuous listening")

    def speak(self, text: str, callback: Callable = None) -> Dict[str, Any]:
        """Speak text with interrupt capability"""
        if not text or not text.strip():
            return {"success": False, "error": "No text provided"}
        
        if self.tts_provider is None:
            return {"success": False, "error": "TTS not available"}
        
        self._stop_speaking.clear()
        self._set_state(VoiceState.SPEAKING)
        
        # Run TTS in thread to not block
        def speak_thread():
            try:
                if self.tts_provider == "edge":
                    self._speak_edge_tts(text)
                elif self.tts_provider == "gtts":
                    self._speak_gtts(text)
                else:
                    self._speak_pyttsx3(text)
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                if self.state == VoiceState.SPEAKING:
                    self._set_state(VoiceState.LISTENING)
                if callback:
                    callback()
        
        self._tts_thread = threading.Thread(target=speak_thread, daemon=True)
        self._tts_thread.start()
        
        return {
            "success": True,
            "message": "Speaking started",
            "text_length": len(text)
        }

    def _speak_edge_tts(self, text: str):
        """Speak using Edge TTS (high quality, async)"""
        import tempfile
        import asyncio
        
        async def generate_audio():
            voice = "en-US-JennyNeural"  # Natural female voice
            communicate = edge_tts.Communicate(text, voice)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_file = f.name
            
            await communicate.save(temp_file)
            return temp_file
        
        try:
            # Generate audio file
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_file = loop.run_until_complete(generate_audio())
            loop.close()
            
            # Play with pygame (interruptible)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for completion or interrupt
            while pygame.mixer.music.get_busy():
                if self._stop_speaking.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
            
            # Cleanup temp file
            try:
                os.unlink(audio_file)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            # Fallback to pyttsx3
            self._speak_pyttsx3(text)

    def _speak_gtts(self, text: str):
        """Speak using gTTS + pygame"""
        try:
            tts = gTTS(text=text, lang='en')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            pygame.mixer.music.load(fp, "mp3")
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                if self._stop_speaking.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
                
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3 (local, less interruptible)"""
        try:
            # Create fresh engine in this thread (Windows requirement)
            engine = pyttsx3.init()
            engine.setProperty('rate', self.tts_rate)
            engine.setProperty('volume', self.tts_volume)
            
            # Set voice
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            # Speak in chunks for better interrupt response
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            for sentence in sentences:
                if self._stop_speaking.is_set():
                    break
                if sentence.strip():
                    engine.say(sentence.strip())
                    engine.runAndWait()
            
            engine.stop()
            
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current voice agent status"""
        return {
            "state": self.state.value,
            "microphone_available": self.microphone is not None,
            "tts_provider": self.tts_provider,
            "is_listening": self.state == VoiceState.LISTENING,
            "is_speaking": self.state == VoiceState.SPEAKING,
            "settings": {
                "language": self.language,
                "tts_rate": self.tts_rate,
                "tts_volume": self.tts_volume
            }
        }

    def cleanup(self):
        """Cleanup resources"""
        self.stop_conversation()
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()


class VoiceConversationManager:
    """
    Manages full voice conversation flow:
    User speaks -> Recognize -> Send to LLM -> Speak response
    With interrupt support at any point
    """

    def __init__(self, therapy_agent, config_path: str = "config.json"):
        self.voice_agent = RealtimeVoiceAgent(config_path)
        self.therapy_agent = therapy_agent
        
        # Set up speech callback
        self.voice_agent.set_speech_callback(self._on_user_speech)
        
        # Conversation state
        self.session_id = None
        self.conversation_active = False
        
        # Callbacks for external integration
        self._on_transcript: Optional[Callable] = None
        self._on_response: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        logger.info("✓ Voice Conversation Manager initialized")

    def start(self, session_id: str = None) -> Dict[str, Any]:
        """Start voice conversation"""
        self.session_id = session_id
        self.conversation_active = True
        return self.voice_agent.start_conversation()

    def stop(self) -> Dict[str, Any]:
        """Stop voice conversation"""
        self.conversation_active = False
        return self.voice_agent.stop_conversation()

    def interrupt(self) -> Dict[str, Any]:
        """Interrupt and listen"""
        return self.voice_agent.interrupt()

    def _on_user_speech(self, text: str):
        """Handle recognized speech from user"""
        if not self.conversation_active:
            return
        
        logger.info(f"Processing user speech: {text}")
        
        # Notify transcript callback
        if self._on_transcript:
            self._on_transcript({
                "type": "user",
                "text": text,
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            # Get response from therapy agent
            # Build context for the agent
            context = {
                "session_id": self.session_id,
                "input_modality": "voice",
                "user_message": text
            }
            
            # Generate response
            response = self.therapy_agent.generate_response(text, context)
            response_text = response.get("response", response.get("message", ""))
            
            if response_text:
                # Notify response callback
                if self._on_response:
                    self._on_response({
                        "type": "assistant",
                        "text": response_text,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Speak the response
                self.voice_agent.speak(response_text)
            
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            if self._on_error:
                self._on_error(str(e))

    def on_transcript(self, callback: Callable):
        """Set callback for transcripts (user and assistant)"""
        self._on_transcript = callback

    def on_response(self, callback: Callable):
        """Set callback for assistant responses"""
        self._on_response = callback

    def on_error(self, callback: Callable):
        """Set callback for errors"""
        self._on_error = callback

    def get_status(self) -> Dict[str, Any]:
        """Get conversation status"""
        status = self.voice_agent.get_status()
        status["conversation_active"] = self.conversation_active
        status["session_id"] = self.session_id
        return status


# Test function
if __name__ == "__main__":
    print("Testing Realtime Voice Agent")
    print("=" * 50)
    
    agent = RealtimeVoiceAgent()
    
    print(f"\nStatus: {agent.get_status()}")
    
    # Test TTS
    print("\nTesting Text-to-Speech...")
    result = agent.speak("Hello! This is a test of the real-time voice system. You can interrupt me anytime by speaking.")
    print(f"TTS Result: {result}")
    
    # Wait for speech to complete
    import time
    time.sleep(5)
    
    print("\nTest complete!")
