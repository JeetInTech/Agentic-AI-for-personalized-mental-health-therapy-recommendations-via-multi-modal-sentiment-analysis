

  A Multimodal AI Therapy System - a comprehensive mental health support application that analyzes and responds to users through:
  - Text chat
  - Voice conversations (speech-to-text and text-to-speech)
  - Video/facial emotion analysis
  - Agentic mode with goal tracking and adaptive learning

  Models & Technologies Used

  1. Text Analysis Models (HuggingFace Transformers)

  - Emotion Detection: j-hartmann/emotion-english-distilroberta-base
  - Sentiment Analysis: cardiffnlp/twitter-roberta-base-sentiment-latest
  - Mental Health Analysis: rabiaqayyum/autotrain-mental-health-analysis
  - Emotion Recognition (General): SamLowe/roberta-base-go_emotions

  2. Large Language Models (LLM)

  - Ollama (Local): llama3.1:8b
  - Groq API (Cloud): llama-3.3-70b-versatile
  - Fallback to rule-based responses when LLMs unavailable

  3. Voice/Audio Models

  - Speech-to-Text: Google Speech Recognition API
  - Text-to-Speech: pyttsx3 (offline, system voices)
  - Speech Emotion: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

  4. Video/Visual Models

  - Face Detection: OpenCV Haar Cascades
  - Facial Emotion Recognition: FER library (deep learning-based)
  - Optional: dlib for advanced face detection

  5. Supporting Libraries

  - PyTorch for model inference
  - OpenCV for video processing
  - librosa for audio analysis
  - Flask for backend API
  - Streamlit for UI (alternative interface)

  Key Features

  - Crisis detection and intervention
  - Multi-turn therapeutic conversations
  - Real-time emotion monitoring via webcam
  - Voice-based interactions
  - Privacy controls and data encryption
  - Goal tracking and progress monitoring
  - Therapeutic technique suggestions (CBT, mindfulness, grounding, etc.)

  The system uses a cascading fallback approach: tries Ollama first → then Groq → then rule-based responses, ensuring it always provides helpful responses     
  even if AI services are unavailable.




