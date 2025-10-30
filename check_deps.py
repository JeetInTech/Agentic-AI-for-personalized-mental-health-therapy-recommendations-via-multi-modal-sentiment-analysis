#!/usr/bin/env python3
"""
Comprehensive dependency checker for Agentic AI Therapy System
"""

import sys

print("=" * 70)
print("Checking Dependencies for Agentic AI Therapy System")
print("=" * 70)

# Core dependencies
core_dependencies = {
    'flask': 'Core - Web framework',
    'flask_cors': 'Core - CORS support',
    'dotenv': 'Core - Environment variables',
}

# AI/ML dependencies
ai_dependencies = {
    'torch': 'AI/ML - PyTorch deep learning',
    'transformers': 'AI/ML - HuggingFace transformers',
    'nltk': 'AI/ML - Natural language toolkit',
    'textblob': 'AI/ML - Text processing',
}

# Voice dependencies
voice_dependencies = {
    'speech_recognition': 'Voice - Speech-to-text',
    'pyttsx3': 'Voice - Text-to-speech',
    'librosa': 'Voice - Audio analysis',
}

# Video dependencies
video_dependencies = {
    'cv2': 'Video - OpenCV',
    'fer': 'Video - Facial emotion recognition',
    'mediapipe': 'Video - Face detection',
}

# LLM dependencies
llm_dependencies = {
    'groq': 'LLM - Groq API',
    'ollama': 'LLM - Ollama integration',
}

# Data processing
data_dependencies = {
    'numpy': 'Data - Numerical computing',
    'pandas': 'Data - Data analysis',
    'sklearn': 'Data - Machine learning utilities',
}

# Optional dependencies
optional_dependencies = {
    'dlib': 'Optional - Advanced face detection',
}

def check_dependency_group(group_name, dependencies):
    """Check a group of dependencies"""
    print(f"\n{group_name}")
    print("-" * 70)

    available = []
    missing = []

    for dep, description in dependencies.items():
        try:
            __import__(dep)
            available.append((dep, description))
            print(f"  [+] {dep:25s} - {description}")
        except ImportError:
            missing.append((dep, description))
            print(f"  [-] {dep:25s} - {description} [MISSING]")

    return available, missing

# Check all dependency groups
all_available = []
all_missing = []

groups = [
    ("CORE DEPENDENCIES", core_dependencies),
    ("AI/ML DEPENDENCIES", ai_dependencies),
    ("VOICE PROCESSING", voice_dependencies),
    ("VIDEO PROCESSING", video_dependencies),
    ("LLM INTEGRATION", llm_dependencies),
    ("DATA PROCESSING", data_dependencies),
    ("OPTIONAL FEATURES", optional_dependencies),
]

for group_name, deps in groups:
    avail, miss = check_dependency_group(group_name, deps)
    all_available.extend(avail)
    all_missing.extend(miss)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"[+] Available: {len(all_available)}")
print(f"[-] Missing:   {len(all_missing)}")
print(f"Python version: {sys.version.split()[0]}")
print(f"Python path: {sys.executable}")

# Installation instructions
if all_missing:
    print("\n" + "=" * 70)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 70)
    print("\nTo install missing dependencies:")
    print("\n  pip install -r requirements.txt")
    print("\nFor specific components:")

    core_missing = [dep for dep, _ in all_missing if dep in core_dependencies]
    voice_missing = [dep for dep, _ in all_missing if dep in voice_dependencies]
    video_missing = [dep for dep, _ in all_missing if dep in video_dependencies]

    if core_missing:
        print("\n  Core components:")
        print(f"    pip install flask flask-cors python-dotenv")

    if voice_missing:
        print("\n  Voice features:")
        print(f"    pip install speechrecognition pyttsx3 librosa")

    if video_missing:
        print("\n  Video features:")
        print(f"    pip install opencv-python fer mediapipe")

    print("\nFor detailed installation guide, see INSTALLATION.md")
else:
    print("\n[SUCCESS] All required dependencies are installed!")
    print("\nYou can now run the application:")
    print("  python app.py")

print("\n" + "=" * 70)
print("Check completed!")
print("=" * 70)