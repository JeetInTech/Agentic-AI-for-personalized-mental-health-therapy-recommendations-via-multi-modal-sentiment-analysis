#!/usr/bin/env python3
"""
Simple dependency checker for multimodal features
"""

import sys

print("Checking Dependencies for Voice and Video Features")
print("=" * 50)

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
        available_deps.append(f"AVAILABLE: {dep} - {description}")
    except ImportError:
        missing_deps.append(f"MISSING: {dep} - {description}")

print("Status:")
for dep in available_deps:
    print(f"  + {dep}")

for dep in missing_deps:
    print(f"  - {dep}")

if missing_deps:
    print(f"\nTo install missing dependencies, run:")
    print("pip install speechrecognition pyttsx3 opencv-python numpy")
    print("Optional (for better emotion recognition): pip install fer dlib")
    print("\nNote: Some packages may require system-level dependencies")
else:
    print("\nAll dependencies are available! Voice and video features should work.")

print(f"\nPython version: {sys.version}")
print("Test completed!")