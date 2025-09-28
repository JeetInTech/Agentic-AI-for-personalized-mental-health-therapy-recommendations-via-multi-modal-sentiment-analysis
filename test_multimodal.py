#!/usr/bin/env python3
"""
Test script for voice and video agents
Tests basic functionality without running the full Flask app
"""

import sys
import os

print("Testing Voice and Video Agents")
print("=" * 50)

# Test voice agent
print("\n1. Testing Voice Agent...")
try:
    from voice_agent import VoiceAgent
    print("   [OK] Voice agent imported successfully")

    # Try to create voice agent
    voice_agent = VoiceAgent()
    print("   [OK] Voice agent created successfully")

    # Test capabilities
    capabilities = voice_agent.get_voice_capabilities()
    print(f"   [OK] Voice capabilities: {capabilities['speech_recognition']['available']} (SR), {capabilities['text_to_speech']['available']} (TTS)")

    # Test system status
    status = voice_agent.test_voice_system()
    print(f"   [OK] Voice system test: {'PASS' if status['overall_status'] else 'FAIL'}")

except ImportError as e:
    print(f"   [ERROR] Import error: {e}")
except Exception as e:
    print(f"   [ERROR] Voice agent error: {e}")

# Test video agent
print("\n2. Testing Video Agent...")
try:
    from video_agent import VideoAgent
    print("   [OK] Video agent imported successfully")

    # Try to create video agent
    video_agent = VideoAgent()
    print("   [OK] Video agent created successfully")

    # Test capabilities
    capabilities = video_agent.get_video_capabilities()
    print(f"   [OK] Video capabilities: {capabilities['face_detection']['available']} (Face), {capabilities['emotion_recognition']['available']} (Emotion)")

    # Test system status
    status = video_agent.test_video_system()
    print(f"   [OK] Video system test: {'PASS' if status['overall_status'] else 'FAIL'}")

except ImportError as e:
    print(f"   [ERROR] Import error: {e}")
except Exception as e:
    print(f"   [ERROR] Video agent error: {e}")

# Check required dependencies
print("\n3. Checking Dependencies...")

dependencies = {
    'speechrecognition': 'Voice input (speech-to-text)',
    'pyttsx3': 'Voice output (text-to-speech)',
    'cv2': 'Video processing (OpenCV)',
    'numpy': 'Numerical processing',
    'fer': 'Facial emotion recognition (optional)',
    'dlib': 'Advanced face detection (optional)'
}

missing_deps = []
available_deps = []

for dep, description in dependencies.items():
    try:
        __import__(dep)
        available_deps.append(f"   [OK] {dep}: {description}")
    except ImportError:
        missing_deps.append(f"   [MISSING] {dep}: {description}")

print("Available dependencies:")
for dep in available_deps:
    print(dep)

if missing_deps:
    print("\nMissing dependencies:")
    for dep in missing_deps:
        print(dep)
    print(f"\nTo install missing dependencies:")
    print("pip install speechrecognition pyttsx3 opencv-python numpy")
    print("Optional: pip install fer dlib")
else:
    print("\n[SUCCESS] All dependencies are available!")

print(f"\nTest completed!")