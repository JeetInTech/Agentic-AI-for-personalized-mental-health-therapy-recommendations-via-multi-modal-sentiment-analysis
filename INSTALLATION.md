# Installation Guide

## Quick Install

```bash
pip install -r requirements.txt
```

## Step-by-Step Installation (Recommended for Windows)

### 1. Core Dependencies

```bash
# Install Flask and web framework
pip install flask==3.0.0 flask-cors==4.0.0 python-dotenv==1.0.0

# Install PyTorch (CPU version)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```

### 2. AI/ML Models

```bash
# Install transformers and dependencies
pip install transformers==4.36.2 sentencepiece==0.1.99 tokenizers==0.15.0 accelerate==0.25.0

# Install NLP tools
pip install nltk==3.8.1 textblob==0.17.1 vaderSentiment==3.3.2
```

### 3. Voice Processing (Windows Special Instructions)

**PyAudio can be tricky on Windows. Use one of these methods:**

**Method 1: Using pipwin (Recommended for Windows)**
```bash
pip install pipwin
pipwin install pyaudio
```

**Method 2: Download prebuilt wheel**
- Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Download the appropriate .whl file for your Python version
- Install: `pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl`

**Then install other voice packages:**
```bash
pip install pyttsx3==2.90 SpeechRecognition==3.10.1
pip install sounddevice==0.4.6 soundfile==0.12.1 librosa==0.10.1
```

### 4. Video & Computer Vision

```bash
pip install opencv-python==4.9.0.80 mediapipe==0.10.9 pillow==10.2.0 fer==22.5.1
```

### 5. LLM Integration

```bash
pip install groq==0.4.2 ollama==0.1.6 openai==1.12.0
```

### 6. Data Processing

```bash
pip install numpy==1.24.3 pandas==2.1.4 scikit-learn==1.3.2 scipy==1.11.4
```

### 7. Visualization & Others

```bash
pip install matplotlib==3.8.2 plotly==5.18.0 seaborn==0.13.1
pip install requests==2.31.0 aiohttp==3.9.1 cryptography==42.0.0
```

## Platform-Specific Instructions

### Windows
- **Python Version:** Use Python 3.10 or 3.11 (3.12 may have compatibility issues)
- **Visual C++ Build Tools:** Required for some packages
  - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Install "Desktop development with C++"

### macOS
```bash
# Install brew packages first
brew install portaudio

# Then install Python packages
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev libsndfile1 ffmpeg

# Then install Python packages
pip install -r requirements.txt
```

## GPU Support (Optional)

### For NVIDIA GPUs with CUDA 12.1:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For Apple Silicon Macs:
```bash
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
```

## Verify Installation

Run the check script:
```bash
python check_deps.py
```

Or test individual components:
```bash
# Test text analysis
python text_analyzer.py

# Test therapy agent
python therapy_agent.py

# Test voice agent
python voice_agent.py

# Test video agent
python video_agent.py

# Test crisis mode
python test_crisis_mode.py
```

## Common Issues & Solutions

### Issue: PyAudio installation fails
**Solution:** Use pipwin (Windows) or install system dependencies (Linux/macOS)

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue: OpenCV camera not working
**Solution:**
- Check camera permissions in Windows Settings
- Try different camera index in config.json (0, 1, 2)

### Issue: pyttsx3 voice not working
**Solution:**
- Windows: Make sure SAPI5 voices are installed
- macOS: Uses NSSpeechSynthesizer (built-in)
- Linux: Install espeak: `sudo apt-get install espeak`

### Issue: Groq API not working
**Solution:**
- Add your Groq API key to config.json or .env file:
```
GROQ_API_KEY=your_api_key_here
```

### Issue: Ollama not connecting
**Solution:**
- Make sure Ollama is installed and running
- Check if service is running: `ollama serve`
- Test connection: `ollama list`

## Minimal Installation (Core Only)

If you want just the basic text therapy without voice/video:

```bash
# Core
pip install flask flask-cors python-dotenv

# AI/ML
pip install torch transformers sentencepiece

# Text Processing
pip install nltk textblob vaderSentiment

# LLM
pip install groq ollama

# Data
pip install numpy pandas requests
```

## Production Deployment

For production deployment, also install:
```bash
pip install gunicorn==21.2.0
# or for Windows:
pip install waitress==3.0.0
```

## Updating Dependencies

To update all packages to latest compatible versions:
```bash
pip install --upgrade -r requirements.txt
```

## Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Post-Installation Setup

1. **Download NLTK data:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

2. **Configure API keys:**
- Copy `config.json.example` to `config.json`
- Add your Groq API key
- Configure Ollama URL if needed

3. **Test the installation:**
```bash
python app.py
# Visit: http://localhost:5000
```

## Need Help?

- Check logs for detailed error messages
- Visit project documentation
- Create an issue on GitHub with error details
