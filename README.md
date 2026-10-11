# 🧠 Agentic AI for Personalized Mental Health Therapy

## Multi-Modal Sentiment Analysis & Recommendation System

A comprehensive AI-powered mental health support system that combines text, voice, and video analysis to provide personalized therapeutic recommendations and crisis intervention support.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [AI Models & LLMs Used](#ai-models--llms-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Crisis Counselling Mode](#crisis-counselling-mode)
- [API Endpoints](#api-endpoints)
- [Privacy & Security](#privacy--security)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🌟 Overview

This system leverages multiple AI models and modalities (text, voice, video) to provide:
- **Real-time emotion detection** from facial expressions
- **Sentiment analysis** from text and voice inputs
- **Crisis detection and intervention** with immediate resource provision
- **Personalized therapeutic recommendations** based on user patterns
- **Goal tracking and progress monitoring**
- **Secure, encrypted user data storage** with user-controlled retention

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Multi-Modal Input Processing**
  - Text chat analysis
  - **Real-time voice conversation mode** (ChatGPT-style)
  - Voice recognition and synthesis
  - **Real-time video emotion detection with autonomous monitoring** 🤖
  
- **Advanced AI Analysis**
  - Emotion recognition (7 emotions: happy, sad, angry, fear, surprise, disgust, neutral)
  - Sentiment analysis (positive, negative, neutral)
  - Crisis risk assessment
  - **Autonomous pattern detection and intervention** (NEW) 🤖
  - Mental health topic detection

- **Therapeutic Support**
  - CBT (Cognitive Behavioral Therapy) techniques
  - Behavioral activation strategies
  - Mindfulness exercises
  - Coping strategy recommendations

- **Privacy-First Design**
  - User-controlled data retention (1 week to 1 year)
  - End-to-end encryption
  - Local data storage
  - Anonymous or identified sessions

- **Crisis Intervention**
  - Real-time crisis detection
  - Immediate safety resource display
  - 24/7 hotline information (India-specific)
  - Automated alert system

### 🎙️ Voice Conversation Mode (NEW)
- **ChatGPT-Style Voice Interface**
  - Continuous real-time conversation
  - Web Speech API integration
  - Dynamic UI with visual feedback
  - Interrupt capability - speak anytime to interrupt AI
  
- **Intelligent Voice Features**
  - Automatic speech recognition (browser-based)
  - Natural text-to-speech with calming voice selection
  - Visual orb animation showing listening/speaking states
  - Session state management (idle, listening, speaking, processing)
  
- **Seamless User Experience**
  - Single dynamic button interface
  - Black & white minimalist theme
  - Response text display during conversation
  - Welcome greeting on session start
  - Background chat logging for reference

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (index.html)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Text Chat  │  │   Voice I/O │  │  Video Feed │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Flask Backend (app.py)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Session Manager & Router                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────┬─────────┬─────────┬─────────┬────────┬─────────────┘
         │         │         │         │        │
         ▼         ▼         ▼         ▼        ▼
    ┌────────┐ ┌──────┐ ┌───────┐ ┌───────┐ ┌─────────┐
    │  Text  │ │Voice │ │ Video │ │Therapy│ │ Agentic │
    │Analyzer│ │Agent │ │ Agent │ │ Agent │ │ System  │
    └────────┘ └──────┘ └───────┘ └───────┘ └─────────┘
         │         │         │         │         │
         ▼         ▼         ▼         ▼         ▼
    ┌─────────────────────────────────────────────────┐
    │           AI Models & Services                   │
    │  • HuggingFace Transformers                     │
    │  • Groq LLM API (Llama 3.3 70B)                │
    │  • Ollama (Llama 3.1 8B - Local)               │
    │  • OpenCV + FER (Emotion Detection)            │
    │  • Speech Recognition + pyttsx3                │
    └─────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Encrypted Storage    │
              │  • SQLite Database     │
              │  • JSON Session Data   │
              │  • User Goals/Progress │
              └────────────────────────┘
```

---

## 🤖 AI Models & LLMs Used

### 1. **Large Language Models (LLMs)**

#### Primary LLM - Groq API
- **File**: `therapy_agent.py`
- **Model**: `llama-3.3-70b-versatile`
- **Purpose**: Therapeutic conversation generation, empathetic responses
- **Provider**: Groq Cloud API
- **Configuration**: 
  ```python
  model = "llama-3.3-70b-versatile"
  temperature = 0.7
  max_tokens = 300
  ```

#### Fallback LLM - Ollama (Local)
- **File**: `therapy_agent.py`
- **Model**: `llama3.1:8b`
- **Purpose**: Local LLM processing when Groq is unavailable
- **Provider**: Ollama (self-hosted)
- **URL**: `http://localhost:11434`

#### Crisis-Specific LLM
- **File**: `crisis_counselling_mode.py`
- **Model**: `llama-3.3-70b-versatile` (Groq)
- **Purpose**: Crisis intervention, safety planning, de-escalation
- **Special Features**: Enhanced crisis detection, resource recommendation

---

### 2. **Emotion Detection Models**

#### Text Emotion Analysis
- **File**: `text_analyzer.py`
- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Provider**: HuggingFace Transformers
- **Emotions Detected**: 7 classes (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Architecture**: DistilRoBERTa
- **Use Case**: Analyzing emotional content in user text messages

#### Video Facial Emotion Recognition (Autonomous Agent 🤖)
- **File**: `video_agent.py`
- **Library**: `FER` (Facial Emotion Recognition)
- **Model**: Deep Neural Network with MTCNN face detection
- **Emotions Detected**: 7 emotions (happy, sad, angry, fear, surprise, disgust, neutral)
- **Framework**: TensorFlow/Keras backend
- **Features**:
  - Real-time face detection
  - Continuous monitoring mode
  - Confidence scoring
  - Emotion trend analysis
  - **🤖 Autonomous decision-making** (NEW)
  - **🚨 Automatic crisis escalation** (NEW)
  - **📊 Pattern detection** (NEW)

**Autonomous Capabilities:**
- **Emotion Persistence Tracking**: Monitors duration of each emotion in real-time
- **Automatic Escalation Rules**:
  - **SAD**: 5 minutes → Crisis counselor escalation (HIGH priority)
  - **ANGRY**: 4 minutes → Crisis counselor escalation (HIGH priority)
  - **FEAR**: 3 minutes → Crisis counselor escalation (HIGH priority)
  - **DISGUST**: 5 minutes → Therapy agent notification (MEDIUM priority)
  - **NEUTRAL**: 10 minutes → User engagement check (LOW priority)
- **Pattern Detection**: Detects rapid emotional fluctuation and suggests grounding techniques
- **Crisis Integration**: Automatically activates crisis counselor when concerning patterns detected
- **Cooldown Mechanism**: Prevents duplicate escalations within 2-minute intervals

---

### 3. **Sentiment Analysis Models**

#### Twitter-RoBERTa Sentiment
- **File**: `text_analyzer.py`
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Provider**: HuggingFace Transformers
- **Sentiments**: Negative, Neutral, Positive
- **Architecture**: RoBERTa-base
- **Use Case**: General sentiment detection in conversations

#### VADER Sentiment (Rule-based)
- **File**: `text_analyzer.py`
- **Library**: `vaderSentiment`
- **Type**: Lexicon and rule-based sentiment analyzer
- **Output**: Compound score (-1 to +1)
- **Use Case**: Backup sentiment analysis, social media text

---

### 4. **Voice Processing Models**

#### Speech Recognition
- **File**: `voice_agent.py`
- **Library**: `SpeechRecognition`
- **Engine**: Google Speech Recognition API
- **Language**: en-US (configurable)
- **Features**: Ambient noise adjustment, timeout handling

#### Text-to-Speech
- **File**: `voice_agent.py`
- **Library**: `pyttsx3`
- **Engine**: Platform-specific (SAPI5 on Windows, nsss on macOS)
- **Voices**: Male/Female configurable
- **Settings**: Rate: 180 WPM, Volume: 0.8

---

### 5. **Computer Vision Models**

#### Face Detection
- **File**: `video_agent.py`
- **Library**: OpenCV
- **Model**: Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Purpose**: Detecting faces in video frames before emotion analysis

#### Optional: MediaPipe (Future Enhancement)
- **Library**: `mediapipe`
- **Purpose**: Advanced face mesh and landmark detection
- **Status**: Installed but not actively used

---

## 📁 Project Structure

```
health/
├── app.py                          # Main Flask application & API endpoints
├── therapy_agent.py                # LLM integration (Groq, Ollama, fallbacks)
├── text_analyzer.py                # Text emotion & sentiment analysis
├── voice_agent.py                  # Speech recognition & TTS
├── video_agent.py                  # Video emotion detection (FER)
├── agentic_therapy_system.py       # User memory, goals, progress tracking
├── crisis_counselling_mode.py      # Crisis detection & intervention
├── crisis_api.py                   # Crisis-specific API endpoints
├── config.json                     # Configuration file (models, APIs, settings)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
│
├── templates/
│   └── index.html                  # Frontend UI (HTML/CSS/JavaScript)
│
├── static/                         # Static assets (CSS, JS, images)
│
├── models/                         # Cached HuggingFace models
├── logs/                           # Application & crisis event logs
├── session_data/                   # Session persistence (JSON)
├── privacy_records/                # User consent records
├── video_data/                     # Video analysis snapshots
├── keys/
│   └── encryption.key              # Encryption key for user data
│
├── user_memory.db                  # SQLite database (encrypted user data)
│
└── tests/
    ├── test_system.py              # Integration tests
    ├── test_emotion_detection.py   # Emotion model tests
    ├── test_voice_output.py        # Voice system tests
    ├── test_crisis_mode.py         # Crisis detection tests
    └── test_multimodal.py          # Multi-modal analysis tests
```

---

## 🔄 System Flow

### 1. **Text Analysis Flow**
```
User Input (Text)
    ↓
text_analyzer.py
    ↓
├─→ Emotion Model (DistilRoBERTa)
├─→ Sentiment Model (Twitter-RoBERTa)
├─→ Crisis Keyword Detection
└─→ Mental Health Topic Detection
    ↓
therapy_agent.py (LLM Processing)
    ↓
├─→ Groq API (Llama 3.3 70B) [Primary]
└─→ Ollama (Llama 3.1 8B) [Fallback]
    ↓
Generated Response + Analysis
    ↓
Frontend Display
```

### 2. **Voice Analysis Flow**
```
User Voice Input
    ↓
voice_agent.py
    ↓
Speech Recognition (Google API)
    ↓
Text Transcription
    ↓
[Same as Text Analysis Flow]
    ↓
TTS Synthesis (pyttsx3)
    ↓
Audio Output
```

### 3. **Video Analysis Flow**
```
Camera Feed
    ↓
video_agent.py
    ↓
OpenCV Face Detection (Haar Cascade)
    ↓
FER Emotion Detection (Deep Neural Network)
    ↓
├─→ Dominant Emotion
├─→ Confidence Score
├─→ All Emotion Probabilities
└─→ Therapeutic Analysis
    ↓
Real-time Display + Trend Tracking
```

### 4. **Crisis Detection Flow**
```
User Input (Any Modality)
    ↓
text_analyzer.py (Crisis Keywords)
    ↓
Crisis Risk Score (0.0 - 1.0)
    ↓
[If Score > 0.5]
    ↓
crisis_counselling_mode.py
    ↓
├─→ Immediate Safety Resources
├─→ Crisis-Specific LLM Response
├─→ Hotline Information (India)
└─→ Event Logging
    ↓
Crisis Alert Display
```

### 5. **Agentic Memory Flow**
```
User Session
    ↓
agentic_therapy_system.py
    ↓
├─→ Encrypt User Data (Fernet)
├─→ Store in SQLite Database
├─→ Track Goals & Progress
├─→ Learn User Patterns
└─→ Generate Personalized Insights
    ↓
Proactive Check-ins & Recommendations
```

---

## 🚀 Installation

### Prerequisites
- **Python**: 3.10 or 3.11 (recommended)
- **Operating System**: Windows, macOS, or Linux
- **Camera**: Optional (for video features)
- **Microphone**: Optional (for voice features)

### Quick Install

```bash
# 1. Clone Repository
git clone https://github.com/JeetInTech/Agentic-AI-for-personalized-mental-health-therapy-recommendations-via-multi-modal-sentiment-analysis.git
cd health

# 2. Create Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Fix NumPy compatibility (Important!)
pip install "numpy<2.0"
pip install "protobuf==4.25.3"

# 4. Install Dependencies
pip install -r requirements.txt
```

### Windows-Specific Instructions

**PyAudio Installation (Required for Voice Features):**
```bash
pip install pipwin
pipwin install pyaudio
```

**Visual C++ Build Tools:**
Some packages require C++ compiler. Download from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/
Install "Desktop development with C++"

### macOS Instructions
```bash
brew install portaudio
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian) Instructions
```bash
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev libsndfile1 ffmpeg
pip install -r requirements.txt
```

### Optional: Install Ollama (Local LLM)
Download from: https://ollama.ai/download

```bash
ollama pull llama3.1:8b
```

### Configure Environment Variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here
```

Get Groq API key from: https://console.groq.com/

### Run the Application
```bash
python app.py
```

Access at: **`http://localhost:5000`**

---

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "models": {
    "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
    "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
  },
  "llm": {
    "groq_model": "llama-3.3-70b-versatile",
    "ollama_model": "llama3.1:8b",
    "temperature": 0.7,
    "max_tokens": 300
  },
  "analysis": {
    "crisis_threshold_high": 0.8,
    "crisis_threshold_moderate": 0.5
  },
  "voice": {
    "tts_rate": 180,
    "tts_volume": 0.8,
    "language": "en-US"
  },
  "video": {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "analysis_interval": 1.0
  }
}
```

---

## 📖 Usage

### 1. **Starting a Session**

**Private Mode** (No memory):
- No data stored
- Each session is independent
- Maximum privacy

**Agentic Mode** (Personalized):
- Data encrypted and stored locally
- Remembers conversations and goals
- Tracks progress over time
- Personalized insights

### 2. **Text Chat**
- Type messages in the chat interface
- Receive AI-powered therapeutic responses
- View real-time emotion and sentiment analysis

### 3. **Voice Input**
- Click the microphone button
- Speak your message
- System transcribes and processes
- Optional: Enable auto-speak for voice responses

### 3a. **Voice Conversation Mode** 🆕
**ChatGPT-Style Real-Time Voice Therapy**

1. **Starting a Voice Session**
   - Click the "Voice Conversation" button (🎙️)
   - Modal opens with animated orb visualizer
   - Click the main microphone button to begin
   - AI greets you with a welcome message

2. **During the Session**
   - **Idle State**: White button with 🎤 - Click to start
   - **Listening State**: Dark button with ⏸️ - AI is listening to you
   - **Processing State**: Dark button with ⏳ - AI is thinking
   - **Speaking State**: Dark button with 🔊 - AI is responding
   
3. **Interrupt Anytime**
   - Simply start speaking while AI talks
   - Click the hand button (✋) to interrupt
   - AI stops immediately and listens to you
   - No need to wait for AI to finish

4. **Session Controls**
   - Main button: Start/Pause conversation
   - Left button: Interrupt AI when speaking
   - Right button: End session (✕)
   
5. **Visual Feedback**
   - Animated orb pulses when listening/speaking
   - Response text shows what AI is saying
   - Status text shows current action
   - All conversations logged in background chat

### 4. **Video Analysis**
- Click "Enable Video"
- Allow camera permissions
- System detects facial emotions in real-time
- View emotion trends over time

### 5. **Goal Tracking**
- Create therapeutic goals
- Track progress
- Receive milestone updates
- Get personalized recommendations

### 6. **Crisis Resources**
- Automatic crisis detection
- Immediate display of helpline numbers
- Coping strategies
- Safety planning resources

---

## 🔌 API Endpoints

### Session Management
```
POST   /api/session/new              # Create new session
POST   /api/privacy/consent/request  # Request privacy consent
POST   /api/privacy/consent/respond  # Submit consent choice
POST   /api/user/authenticate         # Authenticate returning user
```

### Chat & Analysis
```
POST   /api/chat/send                # Send message & get response
GET    /api/providers/status         # Check LLM provider status
POST   /api/analyze/text             # Analyze text only
```

### Voice
```
GET    /api/voice/status             # Check voice system status
POST   /api/voice/listen             # Start voice recognition
POST   /api/voice/speak              # Synthesize speech
```

### Video
```
GET    /api/video/status             # Check video system status
POST   /api/video/start              # Start camera
POST   /api/video/analyze            # Analyze current frame
POST   /api/video/stop               # Stop camera
GET    /api/video/stream             # Video stream endpoint
```

### Crisis
```
POST   /api/crisis/assess            # Assess crisis risk
POST   /api/crisis/escalate          # Escalate to crisis mode
GET    /api/crisis/resources         # Get crisis resources
POST   /api/crisis/safety-plan       # Generate safety plan
```

### Goals
```
POST   /api/goals/create             # Create new goal
GET    /api/goals/list               # List user goals
PUT    /api/goals/update             # Update goal progress
DELETE /api/goals/{id}               # Delete goal
```

---

## 🆘 Crisis Resources (India)

### Emergency Helplines
- **KIRAN Mental Health**: 1800-599-0019 (24/7)
- **Vandrevala Foundation**: 9152987821 (24/7)
- **Emergency Services**: 112

### Additional Resources
- **Suicide Prevention Helpline**: 044-24640050
- **iCall Psychosocial Helpline**: 9152987821
- **NIMHANS Helpline**: 080-46110007

**⚠️ Disclaimer**: This is an AI support tool, not a replacement for professional mental health care. In case of emergency, please call 112 or visit the nearest hospital.

---

## 🔒 Privacy & Security

### Data Protection
- **Encryption**: AES-256 (Fernet) encryption for all user data
- **Storage**: Local SQLite database (no cloud storage)
- **Password Hashing**: PBKDF2 with SHA-256
- **Session Security**: UUID-based session tokens

### User Control
- Choose data retention period (1 week to 1 year)
- Delete account and all data anytime
- Export personal data in JSON format
- Anonymous session option available

### Logging
- Crisis events logged for safety (anonymized)
- User IDs hashed in logs
- Configurable log retention
- GDPR-compliant data handling

---

## 🆘 Crisis Counselling Mode

### Features
- **Automatic Crisis Detection**: Identifies 13+ crisis types (suicide ideation, self-harm, depression, anxiety, grief, etc.)
- **Adaptive Responses**: Emotionally intelligent, context-aware support
- **Evidence-Based Strategies**: Immediate, short-term, and long-term coping techniques
- **Professional Resources**: Crisis hotlines and emergency contacts
- **LLM Integration**: Enhanced empathy through Groq/Ollama with graceful fallbacks

### Testing Crisis Mode
Try these messages:
- **Suicidal Ideation**: "I want to end it all"
- **Depression**: "I feel hopeless and empty"
- **Anxiety**: "I'm having a panic attack"
- **Grief**: "I lost someone I love"

### Crisis Resources (India)
- **KIRAN Mental Health**: 1800-599-0019 (24/7)
- **Vandrevala Foundation**: 9152987821 (24/7)
- **Emergency Services**: 112
- **NIMHANS Helpline**: 080-46110007

---

## 🛠️ Troubleshooting

### Installation Issues

**1. NumPy/Protobuf Version Conflicts**
```bash
pip uninstall numpy protobuf -y
pip install "numpy<2.0" "protobuf==4.25.3"
```

**2. PyAudio Installation Error (Windows)**
```bash
pip install pipwin
pipwin install pyaudio
```

**3. Transformers Import Error**
```bash
pip install --upgrade transformers torch
```

### Runtime Issues

**4. Camera Not Working**
- Check camera permissions in Windows Settings
- Ensure no other app is using the camera
- Try different camera_index in config.json (0, 1, 2)

**5. Voice Recognition Not Working**
- Check microphone permissions
- Test microphone in system settings
- Update audio drivers

**6. LLM Not Responding**
- Verify GROQ_API_KEY in .env file
- Check internet connection
- Ensure Ollama is running (if using local fallback)
- Check logs/app.log for errors

**7. Models Not Downloading**
- Stable internet connection required
- First run downloads ~2-5GB of models
- Check available disk space
- Models cache in `models/` directory

### Chrome-Specific Issues

**8. Consent Popup Not Showing in Chrome**
- Clear browser cache
- Disable browser extensions
- Try incognito mode
- Check browser console for errors (F12)

---

## 📊 System Requirements

- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: ~5GB for models and dependencies
- **CPU**: Multi-core recommended for real-time video
- **GPU**: Optional (CUDA support for faster inference)
- **Python**: 3.10 or 3.11

### Model Loading Times
- Text Models: 5-10 seconds (first load)
- Video Models: 10-15 seconds (first load)
- LLM Response: 2-5 seconds (Groq), 5-15 seconds (Ollama)

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👥 Author

**JeetInTech** - [GitHub](https://github.com/JeetInTech)

---

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Groq for LLM API access
- Ollama for local LLM deployment
- OpenCV and FER for emotion recognition
- Mental health professionals for guidance

---

**⚠️ Medical Disclaimer**

This application provides supportive information only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for mental health concerns. In emergencies, call 112 or visit the nearest hospital immediately.

---

Made with ❤️ for mental health awareness
