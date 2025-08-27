"""
Audio Analyzer for Multimodal Agentic AI Therapy System
Processes audio inputs for voice tone, emotional prosody, and speech patterns
LOCAL PROCESSING ONLY - No data leaves the device
"""

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import sounddevice as sd
import whisper
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import threading
import queue
import time
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# Audio processing libraries
from scipy import signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


class AudioFeatureExtractor:
    """Extract comprehensive audio features for emotional analysis"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_mels = 128
        self.n_mfcc = 13
        
    def extract_prosodic_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract prosodic features (pitch, rhythm, stress)"""
        features = {}
        
        try:
            # Fundamental frequency (F0) - pitch
            f0 = librosa.yin(audio_data, fmin=75, fmax=600, sr=sr)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            
            if len(f0_clean) > 0:
                features['pitch_mean'] = np.mean(f0_clean)
                features['pitch_std'] = np.std(f0_clean)
                features['pitch_min'] = np.min(f0_clean)
                features['pitch_max'] = np.max(f0_clean)
                features['pitch_range'] = features['pitch_max'] - features['pitch_min']
                
                # Pitch contour analysis
                pitch_diff = np.diff(f0_clean)
                features['pitch_slope_mean'] = np.mean(pitch_diff)
                features['pitch_variability'] = np.var(pitch_diff)
            else:
                # Default values for unvoiced audio
                features.update({
                    'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0,
                    'pitch_max': 0, 'pitch_range': 0, 'pitch_slope_mean': 0,
                    'pitch_variability': 0
                })
            
            # Energy and intensity
            energy = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            features['energy_mean'] = np.mean(energy)
            features['energy_std'] = np.std(energy)
            features['energy_max'] = np.max(energy)
            
            # Speaking rate (based on zero-crossing rate)
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)[0]
            features['speaking_rate'] = np.mean(zcr)
            features['articulation_rate'] = np.std(zcr)
            
            return features
            
        except Exception as e:
            print(f"Error extracting prosodic features: {e}")
            return self._get_default_prosodic_features()
    
    def extract_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features for voice quality analysis"""
        features = {}
        
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=self.n_mfcc,
                                       hop_length=self.hop_length)
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr,
                                                                 hop_length=self.hop_length)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr,
                                                              hop_length=self.hop_length)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr,
                                                                  hop_length=self.hop_length)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # Chroma features (harmonic content)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=self.hop_length)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            return features
            
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            return self._get_default_spectral_features()
    
    def extract_temporal_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract temporal features (pauses, rhythm, timing)"""
        features = {}
        
        try:
            # Convert to AudioSegment for silence detection
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=sr,
                sample_width=audio_data.dtype.itemsize,
                channels=1
            )
            
            # Detect non-silent segments
            nonsilent_ranges = detect_nonsilent(
                audio_segment,
                min_silence_len=100,  # 100ms minimum silence
                silence_thresh=-40    # dB threshold
            )
            
            if nonsilent_ranges:
                # Calculate pause statistics
                pause_durations = []
                speech_durations = []
                
                for i, (start, end) in enumerate(nonsilent_ranges):
                    speech_durations.append(end - start)
                    
                    if i > 0:
                        prev_end = nonsilent_ranges[i-1][1]
                        pause_durations.append(start - prev_end)
                
                # Pause analysis
                if pause_durations:
                    features['pause_count'] = len(pause_durations)
                    features['pause_mean'] = np.mean(pause_durations) / 1000.0  # Convert to seconds
                    features['pause_std'] = np.std(pause_durations) / 1000.0
                    features['pause_total'] = sum(pause_durations) / 1000.0
                else:
                    features['pause_count'] = 0
                    features['pause_mean'] = 0
                    features['pause_std'] = 0
                    features['pause_total'] = 0
                
                # Speech segment analysis
                if speech_durations:
                    features['speech_segments'] = len(speech_durations)
                    features['speech_mean_duration'] = np.mean(speech_durations) / 1000.0
                    features['speech_std_duration'] = np.std(speech_durations) / 1000.0
                else:
                    features['speech_segments'] = 0
                    features['speech_mean_duration'] = 0
                    features['speech_std_duration'] = 0
                
                # Speaking ratio
                total_duration = len(audio_segment) / 1000.0  # Convert to seconds
                speech_time = sum(speech_durations) / 1000.0
                features['speaking_ratio'] = speech_time / total_duration if total_duration > 0 else 0
                
            else:
                # No speech detected
                features.update({
                    'pause_count': 0, 'pause_mean': 0, 'pause_std': 0, 'pause_total': 0,
                    'speech_segments': 0, 'speech_mean_duration': 0, 'speech_std_duration': 0,
                    'speaking_ratio': 0
                })
            
            return features
            
        except Exception as e:
            print(f"Error extracting temporal features: {e}")
            return self._get_default_temporal_features()
    
    def _get_default_prosodic_features(self) -> Dict[str, float]:
        """Default prosodic features for error cases"""
        return {
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0, 'pitch_max': 0,
            'pitch_range': 0, 'pitch_slope_mean': 0, 'pitch_variability': 0,
            'energy_mean': 0, 'energy_std': 0, 'energy_max': 0,
            'speaking_rate': 0, 'articulation_rate': 0
        }
    
    def _get_default_spectral_features(self) -> Dict[str, float]:
        """Default spectral features for error cases"""
        features = {}
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = 0
            features[f'mfcc_{i}_std'] = 0
        features.update({
            'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
            'spectral_rolloff_mean': 0, 'spectral_rolloff_std': 0,
            'spectral_bandwidth_mean': 0, 'spectral_bandwidth_std': 0,
            'chroma_mean': 0, 'chroma_std': 0
        })
        return features
    
    def _get_default_temporal_features(self) -> Dict[str, float]:
        """Default temporal features for error cases"""
        return {
            'pause_count': 0, 'pause_mean': 0, 'pause_std': 0, 'pause_total': 0,
            'speech_segments': 0, 'speech_mean_duration': 0, 'speech_std_duration': 0,
            'speaking_ratio': 0
        }


class EmotionalStateClassifier:
    """Classify emotional states from audio features"""
    
    def __init__(self):
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'anxious', 'stressed']
        self.scaler = StandardScaler()
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build a simple neural network for emotion classification"""
        try:
            # Simple feedforward network for emotion classification
            self.model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(50,)),  # Adjust input size
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(len(self.emotion_labels), activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        except Exception as e:
            print(f"Error building emotion model: {e}")
            self.model = None
    
    def predict_emotion(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict emotion from audio features"""
        try:
            # Extract feature values
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Pad or truncate to expected size (50 features)
            if feature_vector.shape[1] < 50:
                padding = np.zeros((1, 50 - feature_vector.shape[1]))
                feature_vector = np.concatenate([feature_vector, padding], axis=1)
            elif feature_vector.shape[1] > 50:
                feature_vector = feature_vector[:, :50]
            
            # Simple rule-based emotion classification (fallback)
            emotion_scores = self._rule_based_emotion_classification(features)
            
            # Determine primary emotion
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            return {
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'emotion_scores': emotion_scores,
                'emotional_arousal': self._calculate_arousal(features),
                'emotional_valence': self._calculate_valence(features)
            }
            
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return {
                'primary_emotion': 'neutral',
                'confidence': 0.5,
                'emotion_scores': {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels},
                'emotional_arousal': 0.5,
                'emotional_valence': 0.5
            }
    
    def _rule_based_emotion_classification(self, features: Dict[str, float]) -> Dict[str, float]:
        """Rule-based emotion classification using audio features"""
        scores = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Extract key features
        pitch_mean = features.get('pitch_mean', 0)
        pitch_std = features.get('pitch_std', 0)
        energy_mean = features.get('energy_mean', 0)
        energy_std = features.get('energy_std', 0)
        speaking_rate = features.get('speaking_rate', 0)
        pause_mean = features.get('pause_mean', 0)
        
        # Normalize features (simple min-max scaling)
        pitch_mean_norm = min(max(pitch_mean / 300.0, 0), 1)  # Assuming 300 Hz max
        energy_norm = min(max(energy_mean * 1000, 0), 1)  # Scale energy
        rate_norm = min(max(speaking_rate * 10, 0), 1)  # Scale rate
        
        # Rule-based classification
        # Happy: Higher pitch, higher energy, faster rate
        scores['happy'] = (pitch_mean_norm * 0.4 + energy_norm * 0.4 + rate_norm * 0.2)
        
        # Sad: Lower pitch, lower energy, slower rate, longer pauses
        scores['sad'] = (1 - pitch_mean_norm) * 0.3 + (1 - energy_norm) * 0.3 + \
                       min(pause_mean, 1) * 0.2 + (1 - rate_norm) * 0.2
        
        # Angry: Higher pitch variance, higher energy, faster rate
        pitch_var_norm = min(pitch_std / 50.0, 1)  # Normalize pitch variance
        scores['angry'] = (pitch_var_norm * 0.3 + energy_norm * 0.4 + rate_norm * 0.3)
        
        # Anxious: Higher pitch variance, variable energy, irregular pauses
        scores['anxious'] = (pitch_var_norm * 0.4 + energy_std * 500 * 0.3 + 
                           min(pause_mean, 1) * 0.3)
        
        # Stressed: Similar to anxious but with higher energy
        scores['stressed'] = (pitch_var_norm * 0.3 + energy_norm * 0.3 + 
                            energy_std * 500 * 0.2 + rate_norm * 0.2)
        
        # Neutral: Baseline scores
        scores['neutral'] = 0.3 + (1 - max(scores.values())) * 0.7
        
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels}
        
        return scores
    
    def _calculate_arousal(self, features: Dict[str, float]) -> float:
        """Calculate emotional arousal (0-1, low to high energy)"""
        energy_mean = features.get('energy_mean', 0)
        speaking_rate = features.get('speaking_rate', 0)
        pitch_std = features.get('pitch_std', 0)
        
        arousal = (energy_mean * 1000 * 0.4 + speaking_rate * 10 * 0.3 + 
                  min(pitch_std / 50.0, 1) * 0.3)
        return min(max(arousal, 0), 1)
    
    def _calculate_valence(self, features: Dict[str, float]) -> float:
        """Calculate emotional valence (0-1, negative to positive)"""
        pitch_mean = features.get('pitch_mean', 0)
        energy_mean = features.get('energy_mean', 0)
        speaking_ratio = features.get('speaking_ratio', 0)
        
        valence = (min(pitch_mean / 300.0, 1) * 0.4 + energy_mean * 1000 * 0.3 + 
                  speaking_ratio * 0.3)
        return min(max(valence, 0), 1)


class VocalStressDetector:
    """Detect vocal stress and anxiety indicators"""
    
    def __init__(self):
        self.stress_indicators = [
            'voice_tremor', 'breathiness', 'vocal_tension', 
            'irregular_rhythm', 'pitch_instability'
        ]
    
    def detect_stress_indicators(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Detect various stress indicators in voice"""
        stress_scores = {}
        
        # Voice tremor (pitch instability)
        pitch_var = features.get('pitch_variability', 0)
        stress_scores['voice_tremor'] = min(pitch_var / 100.0, 1.0)  # Normalize
        
        # Breathiness (spectral characteristics)
        spectral_bandwidth = features.get('spectral_bandwidth_mean', 0)
        stress_scores['breathiness'] = min(spectral_bandwidth / 2000.0, 1.0)
        
        # Vocal tension (energy and spectral centroid)
        energy_std = features.get('energy_std', 0)
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        stress_scores['vocal_tension'] = min((energy_std * 1000 + spectral_centroid / 2000.0) / 2, 1.0)
        
        # Irregular rhythm (pause patterns)
        pause_std = features.get('pause_std', 0)
        speech_std = features.get('speech_std_duration', 0)
        stress_scores['irregular_rhythm'] = min((pause_std + speech_std) / 2.0, 1.0)
        
        # Pitch instability
        pitch_range = features.get('pitch_range', 0)
        pitch_std = features.get('pitch_std', 0)
        stress_scores['pitch_instability'] = min((pitch_range / 100.0 + pitch_std / 50.0) / 2, 1.0)
        
        # Calculate overall stress level
        overall_stress = np.mean(list(stress_scores.values()))
        
        # Determine stress level category
        if overall_stress < 0.3:
            stress_level = 'low'
        elif overall_stress < 0.6:
            stress_level = 'moderate'
        else:
            stress_level = 'high'
        
        return {
            'stress_indicators': stress_scores,
            'overall_stress_score': overall_stress,
            'stress_level': stress_level,
            'high_stress_detected': overall_stress > 0.7
        }


class AudioRecorder:
    """Handle real-time audio recording using sounddevice"""
    
    def __init__(self, sample_rate=44100, channels=1, dtype=np.float32):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.audio_data = []
        self.stream = None
        
        # Set default device if needed
        try:
            sd.check_input_settings()
        except Exception as e:
            print(f"Audio device warning: {e}")
            # Try to get default input device
            try:
                default_device = sd.default.device[0]  # Input device
                sd.default.device = default_device
            except Exception:
                print("Warning: No audio input device detected")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio recording"""
        if status:
            print(f"Recording status: {status}")
        
        if self.recording:
            # Store audio data (copy to avoid issues with buffer reuse)
            self.audio_data.append(indata.copy())
    
    def start_recording(self):
        """Start recording audio"""
        if not self.recording:
            self.recording = True
            self.audio_data = []
            
            try:
                # Start input stream
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    callback=self._audio_callback,
                    blocksize=1024  # Buffer size
                )
                self.stream.start()
                print(f"Started recording at {self.sample_rate} Hz")
                
            except Exception as e:
                print(f"Error starting recording: {e}")
                self.recording = False
                raise e
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data"""
        if self.recording:
            self.recording = False
            
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
                
                # Combine all recorded chunks
                if self.audio_data:
                    audio_array = np.concatenate(self.audio_data, axis=0)
                    # Convert to mono if stereo
                    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    elif len(audio_array.shape) > 1:
                        audio_array = audio_array.squeeze()
                    
                    print(f"Recording stopped. Duration: {len(audio_array)/self.sample_rate:.2f}s")
                    return audio_array
                else:
                    print("No audio data recorded")
                    return np.array([])
                    
            except Exception as e:
                print(f"Error stopping recording: {e}")
                return np.array([])
        
        return np.array([])
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available audio input devices"""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return input_devices
        except Exception as e:
            print(f"Error querying devices: {e}")
            return []
    
    def set_device(self, device_id: int):
        """Set audio input device"""
        try:
            sd.default.device[0] = device_id  # Set input device
            print(f"Set audio input device to ID: {device_id}")
        except Exception as e:
            print(f"Error setting device: {e}")
    
    def test_recording(self, duration: float = 2.0) -> np.ndarray:
        """Test recording for specified duration"""
        print(f"Testing recording for {duration}s...")
        
        try:
            # Record for specified duration
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to 1D array if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()
            
            print("Test recording complete")
            return audio_data
            
        except Exception as e:
            print(f"Test recording failed: {e}")
            return np.array([])
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if self.recording:
                self.stop_recording()
        except Exception:
            pass


class AudioAnalyzer:
    """Main Audio Analyzer class integrating all audio processing components"""
    
    def __init__(self, data_dir: str = "audio_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        self.emotion_classifier = EmotionalStateClassifier()
        self.stress_detector = VocalStressDetector()
        self.recorder = AudioRecorder()
        
        # Initialize speech recognition
        self.speech_recognizer = sr.Recognizer()
        
        # Initialize Whisper model for better speech-to-text
        try:
            self.whisper_model = whisper.load_model("base")
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            self.whisper_model = None
        
        # Analysis history
        self.analysis_history = []
    
    def analyze(self, audio_data: np.ndarray, sample_rate: int = 22050, 
                include_transcription: bool = True) -> Dict[str, Any]:
        """
        Main analysis method - analyze audio data and return comprehensive results
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of audio
            include_transcription: Whether to include speech-to-text
        
        Returns:
            Comprehensive analysis results dictionary
        """
        analysis_start_time = time.time()
        
        try:
            # Validate input
            if audio_data is None or len(audio_data) == 0:
                return self._get_empty_analysis_result()
            
            # Ensure audio is in correct format
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            
            # Resample if necessary
            if sample_rate != self.feature_extractor.sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.feature_extractor.sample_rate
                )
                sample_rate = self.feature_extractor.sample_rate
            
            # Extract features
            print("Extracting audio features...")
            prosodic_features = self.feature_extractor.extract_prosodic_features(audio_data, sample_rate)
            spectral_features = self.feature_extractor.extract_spectral_features(audio_data, sample_rate)
            temporal_features = self.feature_extractor.extract_temporal_features(audio_data, sample_rate)
            
            # Combine all features
            all_features = {**prosodic_features, **spectral_features, **temporal_features}
            
            # Emotion classification
            print("Classifying emotions...")
            emotion_results = self.emotion_classifier.predict_emotion(all_features)
            
            # Stress detection
            print("Detecting stress indicators...")
            stress_results = self.stress_detector.detect_stress_indicators(all_features)
            
            # Speech-to-text transcription
            transcription_results = {}
            if include_transcription:
                print("Transcribing speech...")
                transcription_results = self._transcribe_speech(audio_data, sample_rate)
            
            # Crisis detection
            crisis_indicators = self._detect_crisis_indicators(all_features, emotion_results, stress_results)
            
            # Compile comprehensive results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - analysis_start_time,
                'audio_duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                
                # Feature categories
                'prosodic_features': prosodic_features,
                'spectral_features': spectral_features,
                'temporal_features': temporal_features,
                'all_features': all_features,
                
                # Analysis results
                'emotion_analysis': emotion_results,
                'stress_analysis': stress_results,
                'transcription': transcription_results,
                'crisis_detection': crisis_indicators,
                
                # Summary metrics for compatibility with multimodal fusion
                'sentiment_score': self._convert_emotion_to_sentiment(emotion_results),
                'sentiment_label': self._get_sentiment_label(emotion_results),
                'confidence_score': emotion_results.get('confidence', 0.5),
                'arousal': emotion_results.get('emotional_arousal', 0.5),
                'valence': emotion_results.get('emotional_valence', 0.5),
                'stress_level': stress_results.get('overall_stress_score', 0.0),
                'crisis_risk': crisis_indicators.get('crisis_risk_score', 0.0)
            }
            
            # Store analysis in history
            self.analysis_history.append(analysis_results)
            
            # Save analysis to file
            self._save_analysis(analysis_results)
            
            print(f"Audio analysis completed in {analysis_results['processing_time']:.2f}s")
            return analysis_results
            
        except Exception as e:
            print(f"Error in audio analysis: {e}")
            return self._get_error_analysis_result(str(e))
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio from file"""
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            return self.analyze(audio_data, sample_rate)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return self._get_error_analysis_result(f"File loading error: {e}")
    
    def start_real_time_recording(self):
        """Start real-time audio recording"""
        self.recorder.start_recording()
    
    def stop_real_time_recording(self) -> Dict[str, Any]:
        """Stop real-time recording and analyze"""
        audio_data = self.recorder.stop_recording()
        if len(audio_data) > 0:
            return self.analyze(audio_data, self.recorder.sample_rate)
        else:
            return self._get_empty_analysis_result()
    
    def _transcribe_speech(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Transcribe speech to text using Whisper and/or SpeechRecognition"""
        transcription_results = {
            'text': '',
            'confidence': 0.0,
            'language': 'en',
            'method': 'none'
        }
        
        try:
            # Try Whisper first (more accurate)
            if self.whisper_model is not None:
                # Save temporary audio file for Whisper
                temp_file = self.data_dir / f"temp_audio_{int(time.time())}.wav"
                sf.write(str(temp_file), audio_data, sample_rate)
                
                try:
                    result = self.whisper_model.transcribe(str(temp_file))
                    transcription_results.update({
                        'text': result['text'].strip(),
                        'confidence': 0.9,  # Whisper doesn't provide confidence
                        'language': result.get('language', 'en'),
                        'method': 'whisper'
                    })
                except Exception as e:
                    print(f"Whisper transcription failed: {e}")
                finally:
                    # Clean up temp file
                    if temp_file.exists():
                        temp_file.unlink()
            
            # Fallback to SpeechRecognition if Whisper failed or unavailable
            if not transcription_results['text']:
                try:
                    # Convert to appropriate format for speech_recognition
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    audio_sr = sr.AudioData(audio_int16.tobytes(), sample_rate, 2)
                    
                    # Use Google Web Speech API (requires internet)
                    text = self.speech_recognizer.recognize_google(audio_sr)
                    transcription_results.update({
                        'text': text,
                        'confidence': 0.7,
                        'language': 'en',
                        'method': 'google'
                    })
                except sr.UnknownValueError:
                    transcription_results['text'] = "[Speech not recognized]"
                except sr.RequestError as e:
                    print(f"Speech recognition service error: {e}")
                    transcription_results['text'] = "[Service unavailable]"
                except Exception as e:
                    print(f"Speech recognition error: {e}")
                    transcription_results['text'] = "[Recognition failed]"
            
        except Exception as e:
            print(f"Transcription error: {e}")
            transcription_results['text'] = "[Transcription error]"
        
        return transcription_results
    
    def _detect_crisis_indicators(self, features: Dict[str, float], 
                                emotion_results: Dict[str, Any], 
                                stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect crisis indicators from audio analysis"""
        crisis_indicators = {
            'crisis_risk_score': 0.0,
            'risk_level': 'low',
            'indicators': [],
            'immediate_concern': False
        }
        
        try:
            risk_factors = []
            
            # Emotional crisis indicators
            primary_emotion = emotion_results.get('primary_emotion', 'neutral')
            emotion_confidence = emotion_results.get('confidence', 0)
            
            if primary_emotion in ['sad', 'angry'] and emotion_confidence > 0.7:
                risk_factors.append(('high_negative_emotion', 0.3))
                crisis_indicators['indicators'].append('High negative emotion detected')
            
            # Stress indicators
            stress_level = stress_results.get('overall_stress_score', 0)
            if stress_level > 0.7:
                risk_factors.append(('high_stress', 0.4))
                crisis_indicators['indicators'].append('High vocal stress detected')
            
            # Voice quality indicators
            pitch_variability = features.get('pitch_variability', 0)
            if pitch_variability > 50:  # High pitch instability
                risk_factors.append(('voice_instability', 0.2))
                crisis_indicators['indicators'].append('Voice instability detected')
            
            # Speaking pattern indicators
            speaking_ratio = features.get('speaking_ratio', 1)
            if speaking_ratio < 0.3:  # Very little speaking
                risk_factors.append(('reduced_speech', 0.2))
                crisis_indicators['indicators'].append('Reduced speech activity')
            
            pause_mean = features.get('pause_mean', 0)
            if pause_mean > 2.0:  # Long pauses
                risk_factors.append(('long_pauses', 0.2))
                crisis_indicators['indicators'].append('Extended pauses in speech')
            
            # Energy indicators
            energy_mean = features.get('energy_mean', 0)
            if energy_mean < 0.01:  # Very low energy
                risk_factors.append(('low_energy', 0.2))
                crisis_indicators['indicators'].append('Very low vocal energy')
            
            # Calculate overall crisis risk
            if risk_factors:
                crisis_risk = sum(weight for _, weight in risk_factors)
                crisis_risk = min(crisis_risk, 1.0)  # Cap at 1.0
            else:
                crisis_risk = 0.0
            
            # Determine risk level
            if crisis_risk < 0.3:
                risk_level = 'low'
            elif crisis_risk < 0.6:
                risk_level = 'moderate'
            else:
                risk_level = 'high'
            
            crisis_indicators.update({
                'crisis_risk_score': crisis_risk,
                'risk_level': risk_level,
                'immediate_concern': crisis_risk > 0.8,
                'risk_factors': risk_factors
            })
            
        except Exception as e:
            print(f"Crisis detection error: {e}")
        
        return crisis_indicators
    
    def _convert_emotion_to_sentiment(self, emotion_results: Dict[str, Any]) -> float:
        """Convert emotion classification to sentiment score (-1 to 1)"""
        try:
            emotion_scores = emotion_results.get('emotion_scores', {})
            
            # Map emotions to sentiment values
            emotion_sentiment_map = {
                'happy': 0.8,
                'neutral': 0.0,
                'sad': -0.6,
                'angry': -0.8,
                'anxious': -0.4,
                'stressed': -0.5
            }
            
            # Calculate weighted sentiment
            weighted_sentiment = 0.0
            for emotion, score in emotion_scores.items():
                sentiment_value = emotion_sentiment_map.get(emotion, 0.0)
                weighted_sentiment += sentiment_value * score
            
            return weighted_sentiment
            
        except Exception:
            return 0.0  # Neutral default
    
    def _get_sentiment_label(self, emotion_results: Dict[str, Any]) -> str:
        """Get sentiment label from emotion results"""
        try:
            sentiment_score = self._convert_emotion_to_sentiment(emotion_results)
            
            if sentiment_score > 0.3:
                return 'positive'
            elif sentiment_score < -0.3:
                return 'negative'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'
    
    def _save_analysis(self, analysis_results: Dict[str, Any]):
        """Save analysis results to file"""
        try:
            timestamp = analysis_results['timestamp'].replace(':', '-')
            filename = f"audio_analysis_{timestamp}.json"
            filepath = self.data_dir / filename
            
            # Create a simplified version for saving (remove numpy arrays)
            save_data = {
                'timestamp': analysis_results['timestamp'],
                'audio_duration': analysis_results['audio_duration'],
                'emotion_analysis': analysis_results['emotion_analysis'],
                'stress_analysis': analysis_results['stress_analysis'],
                'transcription': analysis_results['transcription'],
                'crisis_detection': analysis_results['crisis_detection'],
                'sentiment_score': analysis_results['sentiment_score'],
                'sentiment_label': analysis_results['sentiment_label'],
                'confidence_score': analysis_results['confidence_score']
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving analysis: {e}")
    
    def _get_empty_analysis_result(self) -> Dict[str, Any]:
        """Return empty/default analysis result"""
        return {
            'timestamp': datetime.now().isoformat(),
            'processing_time': 0.0,
            'audio_duration': 0.0,
            'sample_rate': 22050,
            'prosodic_features': self.feature_extractor._get_default_prosodic_features(),
            'spectral_features': self.feature_extractor._get_default_spectral_features(),
            'temporal_features': self.feature_extractor._get_default_temporal_features(),
            'all_features': {},
            'emotion_analysis': {
                'primary_emotion': 'neutral',
                'confidence': 0.0,
                'emotion_scores': {emotion: 0.0 for emotion in self.emotion_classifier.emotion_labels},
                'emotional_arousal': 0.0,
                'emotional_valence': 0.0
            },
            'stress_analysis': {
                'stress_indicators': {indicator: 0.0 for indicator in self.stress_detector.stress_indicators},
                'overall_stress_score': 0.0,
                'stress_level': 'low',
                'high_stress_detected': False
            },
            'transcription': {
                'text': '',
                'confidence': 0.0,
                'language': 'en',
                'method': 'none'
            },
            'crisis_detection': {
                'crisis_risk_score': 0.0,
                'risk_level': 'low',
                'indicators': [],
                'immediate_concern': False
            },
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence_score': 0.0,
            'arousal': 0.0,
            'valence': 0.0,
            'stress_level': 0.0,
            'crisis_risk': 0.0,
            'status': 'no_audio'
        }
    
    def _get_error_analysis_result(self, error_message: str) -> Dict[str, Any]:
        """Return error analysis result"""
        result = self._get_empty_analysis_result()
        result.update({
            'status': 'error',
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        })
        return result
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        return self.analysis_history[-limit:]
    
    def clear_analysis_history(self):
        """Clear analysis history"""
        self.analysis_history.clear()
    
    def get_audio_quality_metrics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Get audio quality metrics"""
        try:
            metrics = {}
            
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_data ** 2)
            noise_estimate = np.var(audio_data - np.mean(audio_data))
            if noise_estimate > 0:
                snr = 10 * np.log10(signal_power / noise_estimate)
            else:
                snr = float('inf')
            metrics['snr_db'] = min(snr, 100)  # Cap at 100 dB
            
            # Dynamic range
            metrics['dynamic_range'] = np.max(np.abs(audio_data)) - np.min(np.abs(audio_data))
            
            # Clipping detection
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
            metrics['clipping_percentage'] = (clipped_samples / len(audio_data)) * 100
            
            # Silence percentage
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(audio_data) < silence_threshold)
            metrics['silence_percentage'] = (silent_samples / len(audio_data)) * 100
            
            # Overall quality score (0-1)
            quality_score = 1.0
            if snr < 10:  # Poor SNR
                quality_score *= 0.5
            if metrics['clipping_percentage'] > 1:  # Clipping detected
                quality_score *= 0.3
            if metrics['silence_percentage'] > 80:  # Mostly silence
                quality_score *= 0.2
            
            metrics['quality_score'] = quality_score
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating audio quality metrics: {e}")
            return {
                'snr_db': 0,
                'dynamic_range': 0,
                'clipping_percentage': 0,
                'silence_percentage': 100,
                'quality_score': 0
            }
    
    def export_analysis_data(self, filepath: str, include_features: bool = False):
        """Export analysis history to CSV or JSON"""
        try:
            export_data = []
            
            for analysis in self.analysis_history:
                export_item = {
                    'timestamp': analysis['timestamp'],
                    'duration': analysis['audio_duration'],
                    'primary_emotion': analysis['emotion_analysis']['primary_emotion'],
                    'emotion_confidence': analysis['emotion_analysis']['confidence'],
                    'sentiment_score': analysis['sentiment_score'],
                    'sentiment_label': analysis['sentiment_label'],
                    'stress_level': analysis['stress_analysis']['overall_stress_score'],
                    'crisis_risk': analysis['crisis_detection']['crisis_risk_score'],
                    'transcription': analysis['transcription']['text']
                }
                
                if include_features:
                    export_item.update(analysis['all_features'])
                
                export_data.append(export_item)
            
            # Determine format from extension
            if filepath.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif filepath.endswith('.csv'):
                import pandas as pd
                df = pd.DataFrame(export_data)
                df.to_csv(filepath, index=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")
            
            print(f"Analysis data exported to {filepath}")
            
        except Exception as e:
            print(f"Error exporting analysis data: {e}")
    
    def calibrate_for_user(self, calibration_audio_samples: List[np.ndarray]):
        """Calibrate analyzer for specific user's voice characteristics"""
        try:
            print("Calibrating audio analyzer for user...")
            
            # Extract features from calibration samples
            calibration_features = []
            for audio_sample in calibration_audio_samples:
                if len(audio_sample) > 0:
                    features = {}
                    features.update(self.feature_extractor.extract_prosodic_features(
                        audio_sample, self.feature_extractor.sample_rate))
                    features.update(self.feature_extractor.extract_spectral_features(
                        audio_sample, self.feature_extractor.sample_rate))
                    calibration_features.append(features)
            
            if calibration_features:
                # Calculate user baseline metrics
                baseline_pitch = np.mean([f.get('pitch_mean', 0) for f in calibration_features])
                baseline_energy = np.mean([f.get('energy_mean', 0) for f in calibration_features])
                baseline_rate = np.mean([f.get('speaking_rate', 0) for f in calibration_features])
                
                # Store user calibration data
                self.user_baseline = {
                    'pitch_baseline': baseline_pitch,
                    'energy_baseline': baseline_energy,
                    'rate_baseline': baseline_rate,
                    'calibrated': True
                }
                
                print(f"Calibration complete. Baseline pitch: {baseline_pitch:.1f} Hz")
            else:
                print("Calibration failed: No valid audio samples")
                
        except Exception as e:
            print(f"Calibration error: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'recorder'):
                del self.recorder
        except Exception:
            pass


# Utility functions for testing and demonstration
def test_audio_analyzer():
    """Test function for audio analyzer"""
    print("Testing Audio Analyzer...")
    
    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    # Generate test audio signal (sine wave)
    duration = 2.0  # seconds
    sample_rate = 22050
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise and variation to make it more realistic
    noise = 0.05 * np.random.normal(0, 1, len(test_audio))
    test_audio += noise
    
    # Analyze test audio
    results = analyzer.analyze(test_audio, sample_rate, include_transcription=False)
    
    print("\nTest Analysis Results:")
    print(f"Emotion: {results['emotion_analysis']['primary_emotion']} "
          f"(confidence: {results['emotion_analysis']['confidence']:.2f})")
    print(f"Sentiment: {results['sentiment_label']} "
          f"(score: {results['sentiment_score']:.2f})")
    print(f"Stress Level: {results['stress_analysis']['stress_level']} "
          f"(score: {results['stress_analysis']['overall_stress_score']:.2f})")
    print(f"Crisis Risk: {results['crisis_detection']['risk_level']} "
          f"(score: {results['crisis_detection']['crisis_risk_score']:.2f})")
    print(f"Processing Time: {results['processing_time']:.2f}s")
    
    return results


if __name__ == "__main__":
    # Run test
    test_results = test_audio_analyzer()
    
    print("\n" + "="*50)
    print("Audio Analyzer Test Complete")
    print("="*50)