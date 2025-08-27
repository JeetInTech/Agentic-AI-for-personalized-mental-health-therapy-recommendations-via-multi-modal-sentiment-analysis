"""
Sentiment Analyzer - Advanced Text Analysis Engine for Therapy
Comprehensive text processing with sentiment analysis, emotion detection,
crisis identification, and therapeutic keyword recognition.
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Text processing libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Advanced NLP libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

# Machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Deep learning for advanced analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Using traditional NLP methods only.")

# Download required NLTK data
def download_nltk_requirements():
    """Download necessary NLTK datasets"""
    required_data = [
        'punkt', 'stopwords', 'wordnet', 'vader_lexicon', 
        'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
    ]
    
    for data in required_data:
        try:
            nltk.download(data, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {data}: {e}")

# Initialize NLTK requirements
download_nltk_requirements()


class SentimentAnalyzer:
    """
    Advanced sentiment analysis system for therapeutic text processing
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.setup_logging()
        self.initialize_analyzers()
        self.load_therapeutic_lexicons()
        self.setup_crisis_detection()
        self.load_or_train_models()
        self.initialize_transformers()
        
        # Analysis history for personalization
        self.analysis_history = []
        self.user_patterns = {}
        
    def setup_logging(self):
        """Setup logging for analysis tracking"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_dir / "sentiment_analysis.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_analyzers(self):
        """Initialize various sentiment analyzers"""
        try:
            # NLTK VADER analyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Alternative VADER analyzer
            self.vader_alt = VaderAnalyzer()
            
            # TextBlob analyzer
            self.textblob_analyzer = TextBlob
            
            # Initialize lemmatizer and tokenizers
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            self.logger.info("Sentiment analyzers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing analyzers: {e}")
            raise
    
    def load_therapeutic_lexicons(self):
        """Load therapeutic and mental health specific lexicons"""
        
        # Emotion categories for therapy
        self.emotion_categories = {
            'positive': {
                'joy': ['happy', 'joyful', 'elated', 'cheerful', 'delighted', 'content', 'pleased', 'glad'],
                'gratitude': ['grateful', 'thankful', 'appreciative', 'blessed', 'fortunate'],
                'hope': ['hopeful', 'optimistic', 'confident', 'encouraged', 'motivated'],
                'love': ['love', 'affection', 'caring', 'warmth', 'compassion', 'empathy'],
                'peace': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'centered']
            },
            'negative': {
                'sadness': ['sad', 'depressed', 'melancholy', 'sorrowful', 'grief', 'mourning', 'heartbroken'],
                'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'overwhelmed', 'panicked', 'tense'],
                'anger': ['angry', 'furious', 'rage', 'irritated', 'frustrated', 'annoyed', 'hostile'],
                'fear': ['afraid', 'scared', 'terrified', 'frightened', 'paranoid', 'phobic'],
                'shame': ['ashamed', 'guilty', 'embarrassed', 'humiliated', 'regretful'],
                'loneliness': ['lonely', 'isolated', 'abandoned', 'disconnected', 'alone', 'rejected']
            }
        }
        
        # Mental health indicators
        self.mental_health_indicators = {
            'depression_indicators': [
                'worthless', 'hopeless', 'empty', 'numb', 'exhausted', 'drained',
                'unmotivated', 'purposeless', 'burden', 'pointless', 'dark', 'heavy'
            ],
            'anxiety_indicators': [
                'racing thoughts', 'cant breathe', 'chest tight', 'spinning', 'spiraling',
                'catastrophizing', 'what if', 'worst case', 'out of control'
            ],
            'trauma_indicators': [
                'flashback', 'triggered', 'nightmare', 'hypervigilant', 'dissociate',
                'numb', 'detached', 'reliving', 'intrusive thoughts'
            ],
            'self_harm_indicators': [
                'cutting', 'burning', 'hurting myself', 'self harm', 'punishing myself',
                'deserve pain', 'release the pain'
            ]
        }
        
        # Therapeutic progress indicators
        self.progress_indicators = {
            'positive_coping': [
                'breathing exercises', 'meditation', 'journaling', 'therapy',
                'support group', 'self care', 'boundaries', 'coping strategies'
            ],
            'insight': [
                'i realize', 'i understand', 'i see now', 'patterns', 'triggers',
                'awareness', 'learning about myself', 'growth'
            ],
            'resilience': [
                'getting through', 'surviving', 'strength', 'resilient',
                'bounce back', 'overcome', 'persevere'
            ]
        }
        
        self.logger.info("Therapeutic lexicons loaded successfully")
    
    def setup_crisis_detection(self):
        """Setup crisis detection keywords and patterns"""
        
        # Crisis keywords with severity weights
        self.crisis_keywords = {
            'suicide': {
                'high_risk': [
                    'kill myself', 'end my life', 'suicide', 'suicidal', 
                    'want to die', 'better off dead', 'no point living',
                    'planning to die', 'goodbye cruel world'
                ],
                'medium_risk': [
                    'dont want to be here', 'tired of living', 'cant go on',
                    'everyone would be better', 'nothing to live for',
                    'wish i was dead', 'life is pointless'
                ],
                'low_risk': [
                    'sometimes think about death', 'wonder about dying',
                    'death seems peaceful', 'tired of everything'
                ]
            },
            'self_harm': {
                'high_risk': [
                    'going to hurt myself', 'cutting tonight', 'punish myself',
                    'deserve the pain', 'need to cut', 'burning myself'
                ],
                'medium_risk': [
                    'want to hurt myself', 'urge to cut', 'thinking about cutting',
                    'need pain', 'self destructive'
                ],
                'low_risk': [
                    'used to cut', 'history of self harm', 'sometimes hurt myself'
                ]
            },
            'violence': {
                'high_risk': [
                    'going to hurt someone', 'kill them', 'they deserve to die',
                    'planning violence', 'get revenge'
                ],
                'medium_risk': [
                    'want to hurt', 'angry enough to kill', 'violent thoughts',
                    'they should pay'
                ],
                'low_risk': [
                    'sometimes angry', 'frustrated with people'
                ]
            }
        }
        
        # Crisis patterns (regex)
        self.crisis_patterns = [
            r'\b(kill|hurt|end)\s+(myself|my\s+life)\b',
            r'\b(suicide|suicidal)\b',
            r'\b(better\s+off\s+dead|want\s+to\s+die)\b',
            r'\b(no\s+point|nothing\s+to\s+live\s+for)\b',
            r'\b(cutting|burning|hurting)\s+(myself|me)\b'
        ]
        
        self.logger.info("Crisis detection system initialized")
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        
        # Paths for model files
        self.emotion_model_path = self.model_dir / "emotion_classifier.pkl"
        self.crisis_model_path = self.model_dir / "crisis_detector.pkl"
        self.vectorizer_path = self.model_dir / "tfidf_vectorizer.pkl"
        
        try:
            # Try loading existing models
            if self.emotion_model_path.exists():
                self.emotion_classifier = joblib.load(self.emotion_model_path)
                self.logger.info("Emotion classifier loaded")
            else:
                self.train_emotion_classifier()
            
            if self.crisis_model_path.exists():
                self.crisis_detector = joblib.load(self.crisis_model_path)
                self.logger.info("Crisis detector loaded")
            else:
                self.train_crisis_detector()
            
            if self.vectorizer_path.exists():
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.logger.info("TF-IDF vectorizer loaded")
            else:
                self.vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
        except Exception as e:
            self.logger.error(f"Error loading/training models: {e}")
            # Fallback: create simple models
            self.create_fallback_models()
    
    def train_emotion_classifier(self):
        """Train emotion classification model"""
        
        # Create synthetic training data based on lexicons
        training_data = []
        labels = []
        
        for category, emotions in self.emotion_categories.items():
            for emotion, words in emotions.items():
                for word in words:
                    # Create sample sentences
                    training_data.extend([
                        f"I feel {word} today",
                        f"I am so {word}",
                        f"Feeling really {word} right now",
                        f"This makes me {word}"
                    ])
                    labels.extend([emotion] * 4)
        
        # Train classifier
        if training_data:
            self.emotion_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            
            self.emotion_classifier.fit(training_data, labels)
            joblib.dump(self.emotion_classifier, self.emotion_model_path)
            self.logger.info("Emotion classifier trained and saved")
        else:
            self.create_fallback_models()
    
    def train_crisis_detector(self):
        """Train crisis detection model"""
        
        # Create training data for crisis detection
        crisis_data = []
        crisis_labels = []
        
        # Positive examples (crisis)
        for category, risk_levels in self.crisis_keywords.items():
            for risk_level, phrases in risk_levels.items():
                weight = {'high_risk': 1, 'medium_risk': 1, 'low_risk': 1}[risk_level]
                for phrase in phrases:
                    crisis_data.extend([
                        phrase,
                        f"I am {phrase}",
                        f"Sometimes {phrase}",
                        f"Thinking about {phrase}"
                    ])
                    crisis_labels.extend([1] * 4)  # 1 = crisis
        
        # Negative examples (non-crisis)
        normal_phrases = [
            "I feel good today", "Having a great day", "Things are going well",
            "I'm happy", "Life is good", "Feeling positive", "Great mood",
            "Excited about tomorrow", "Love spending time with friends"
        ]
        
        for phrase in normal_phrases:
            crisis_data.extend([
                phrase,
                f"I am {phrase}",
                f"Really {phrase}",
                f"Feeling {phrase}"
            ])
            crisis_labels.extend([0] * 4)  # 0 = not crisis
        
        # Train crisis detector
        if crisis_data:
            self.crisis_detector = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', LogisticRegression())
            ])
            
            self.crisis_detector.fit(crisis_data, crisis_labels)
            joblib.dump(self.crisis_detector, self.crisis_model_path)
            self.logger.info("Crisis detector trained and saved")
        else:
            self.create_fallback_models()
    
    def create_fallback_models(self):
        """Create simple fallback models if training fails"""
        self.emotion_classifier = None
        self.crisis_detector = None
        self.logger.warning("Using fallback models (keyword-based only)")
    
    def initialize_transformers(self):
        """Initialize transformer models if available"""
        
        if not TRANSFORMERS_AVAILABLE:
            self.transformer_sentiment = None
            self.transformer_emotion = None
            return
        
        try:
            # Initialize sentiment analysis pipeline
            self.transformer_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Initialize emotion analysis pipeline
            self.transformer_emotion = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                tokenizer="j-hartmann/emotion-english-distilroberta-base"
            )
            
            self.logger.info("Transformer models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize transformers: {e}")
            self.transformer_sentiment = None
            self.transformer_emotion = None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common contractions
        contractions = {
            "i'm": "i am", "you're": "you are", "it's": "it is",
            "we're": "we are", "they're": "they are", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "won't": "will not", "wouldn't": "would not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not",
            "mightn't": "might not", "mustn't": "must not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract various linguistic features from text"""
        
        processed_text = self.preprocess_text(text)
        
        features = {
            'word_count': len(processed_text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': np.mean([len(word) for word in processed_text.split()]) if processed_text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'punctuation_density': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        }
        
        # POS tagging features
        try:
            tokens = word_tokenize(processed_text)
            pos_tags = pos_tag(tokens)
            
            # Count different POS types
            pos_counts = {'noun': 0, 'verb': 0, 'adjective': 0, 'adverb': 0, 'pronoun': 0}
            
            for word, pos in pos_tags:
                if pos.startswith('NN'):  # Nouns
                    pos_counts['noun'] += 1
                elif pos.startswith('VB'):  # Verbs
                    pos_counts['verb'] += 1
                elif pos.startswith('JJ'):  # Adjectives
                    pos_counts['adjective'] += 1
                elif pos.startswith('RB'):  # Adverbs
                    pos_counts['adverb'] += 1
                elif pos.startswith('PRP'):  # Pronouns
                    pos_counts['pronoun'] += 1
            
            features.update(pos_counts)
            
        except Exception as e:
            self.logger.warning(f"POS tagging failed: {e}")
        
        return features
    
    def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze emotions in text using multiple methods"""
        
        emotions = {
            'detected_emotions': [],
            'emotion_scores': {},
            'dominant_emotion': 'neutral',
            'emotion_confidence': 0.0
        }
        
        processed_text = self.preprocess_text(text)
        
        # Lexicon-based emotion detection
        for category, emotion_dict in self.emotion_categories.items():
            for emotion, keywords in emotion_dict.items():
                score = sum(1 for keyword in keywords if keyword in processed_text)
                if score > 0:
                    emotions['detected_emotions'].append(emotion)
                    emotions['emotion_scores'][emotion] = score
        
        # Machine learning emotion classification
        if self.emotion_classifier:
            try:
                emotion_pred = self.emotion_classifier.predict([text])[0]
                emotion_proba = max(self.emotion_classifier.predict_proba([text])[0])
                
                emotions['ml_emotion'] = emotion_pred
                emotions['ml_confidence'] = emotion_proba
                
            except Exception as e:
                self.logger.warning(f"ML emotion classification failed: {e}")
        
        # Transformer-based emotion analysis
        if self.transformer_emotion:
            try:
                emotion_result = self.transformer_emotion(text)[0]
                emotions['transformer_emotion'] = emotion_result['label']
                emotions['transformer_confidence'] = emotion_result['score']
                
            except Exception as e:
                self.logger.warning(f"Transformer emotion analysis failed: {e}")
        
        # Determine dominant emotion
        if emotions['emotion_scores']:
            dominant = max(emotions['emotion_scores'], key=emotions['emotion_scores'].get)
            emotions['dominant_emotion'] = dominant
            emotions['emotion_confidence'] = emotions['emotion_scores'][dominant]
        
        return emotions
    
    def detect_crisis(self, text: str) -> Dict[str, Any]:
        """Comprehensive crisis detection"""
        
        crisis_analysis = {
            'crisis_risk': 0.0,
            'risk_level': 'low',
            'crisis_indicators': [],
            'matched_keywords': [],
            'risk_factors': {},
            'immediate_concern': False
        }
        
        processed_text = self.preprocess_text(text)
        
        # Keyword-based detection
        total_risk_score = 0
        max_individual_risk = 0
        
        for category, risk_levels in self.crisis_keywords.items():
            category_risk = 0
            category_matches = []
            
            for risk_level, keywords in risk_levels.items():
                risk_weight = {'high_risk': 0.9, 'medium_risk': 0.6, 'low_risk': 0.3}[risk_level]
                
                for keyword in keywords:
                    if keyword in processed_text:
                        category_matches.append(keyword)
                        category_risk = max(category_risk, risk_weight)
                        crisis_analysis['matched_keywords'].append({
                            'keyword': keyword,
                            'category': category,
                            'risk_level': risk_level,
                            'weight': risk_weight
                        })
            
            if category_matches:
                crisis_analysis['crisis_indicators'].append(category)
                crisis_analysis['risk_factors'][category] = {
                    'risk_score': category_risk,
                    'matched_terms': category_matches
                }
                
                total_risk_score += category_risk
                max_individual_risk = max(max_individual_risk, category_risk)
        
        # Pattern-based detection
        pattern_matches = 0
        for pattern in self.crisis_patterns:
            if re.search(pattern, processed_text, re.IGNORECASE):
                pattern_matches += 1
                total_risk_score += 0.4
        
        # Machine learning crisis detection
        if self.crisis_detector:
            try:
                ml_prediction = self.crisis_detector.predict([text])[0]
                ml_probability = self.crisis_detector.predict_proba([text])[0][1]  # Probability of crisis
                
                crisis_analysis['ml_prediction'] = bool(ml_prediction)
                crisis_analysis['ml_probability'] = ml_probability
                
                # Incorporate ML prediction into overall risk
                total_risk_score += ml_probability * 0.5
                
            except Exception as e:
                self.logger.warning(f"ML crisis detection failed: {e}")
        
        # Calculate final risk score (normalized)
        crisis_analysis['crisis_risk'] = min(total_risk_score / 2.0, 1.0)  # Normalize to 0-1
        
        # Determine risk level
        if crisis_analysis['crisis_risk'] >= 0.8 or max_individual_risk >= 0.9:
            crisis_analysis['risk_level'] = 'high'
            crisis_analysis['immediate_concern'] = True
        elif crisis_analysis['crisis_risk'] >= 0.5:
            crisis_analysis['risk_level'] = 'medium'
        elif crisis_analysis['crisis_risk'] >= 0.2:
            crisis_analysis['risk_level'] = 'elevated'
        else:
            crisis_analysis['risk_level'] = 'low'
        
        return crisis_analysis
    
    def analyze_therapeutic_progress(self, text: str) -> Dict[str, Any]:
        """Analyze therapeutic progress indicators"""
        
        progress = {
            'progress_indicators': [],
            'coping_skills': [],
            'insight_markers': [],
            'resilience_signs': [],
            'progress_score': 0.0
        }
        
        processed_text = self.preprocess_text(text)
        
        # Check for progress indicators
        total_progress = 0
        
        for category, keywords in self.progress_indicators.items():
            matches = [keyword for keyword in keywords if keyword in processed_text]
            if matches:
                progress['progress_indicators'].append(category)
                progress[f'{category}_matches'] = matches
                
                if category == 'positive_coping':
                    progress['coping_skills'].extend(matches)
                    total_progress += len(matches) * 0.3
                elif category == 'insight':
                    progress['insight_markers'].extend(matches)
                    total_progress += len(matches) * 0.4
                elif category == 'resilience':
                    progress['resilience_signs'].extend(matches)
                    total_progress += len(matches) * 0.3
        
        progress['progress_score'] = min(total_progress, 1.0)
        
        return progress
    
    def analyze(self, text: str, user_id: str = None) -> Dict[str, Any]:
        """
        Main analysis function - comprehensive text analysis
        """
        
        if not text or not isinstance(text, str):
            return self.create_empty_analysis()
        
        analysis_start_time = datetime.now()
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Initialize analysis results
            analysis = {
                'timestamp': analysis_start_time,
                'original_text': text,
                'processed_text': processed_text,
                'text_length': len(text),
                'word_count': len(text.split())
            }
            
            # Basic sentiment analysis (VADER)
            vader_scores = self.vader_analyzer.polarity_scores(text)
            analysis['vader_sentiment'] = vader_scores
            
            # Alternative VADER
            vader_alt_scores = self.vader_alt.polarity_scores(text)
            analysis['vader_alt_sentiment'] = vader_alt_scores
            
            # TextBlob sentiment
            blob = self.textblob_analyzer(text)
            analysis['textblob_sentiment'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
            # Transformer-based sentiment (if available)
            if self.transformer_sentiment:
                try:
                    transformer_result = self.transformer_sentiment(text)[0]
                    analysis['transformer_sentiment'] = {
                        'label': transformer_result['label'],
                        'score': transformer_result['score']
                    }
                except Exception as e:
                    self.logger.warning(f"Transformer sentiment analysis failed: {e}")
            
            # Extract linguistic features
            analysis['features'] = self.extract_features(text)
            
            # Emotion analysis
            analysis['emotions'] = self.analyze_emotions(text)
            
            # Crisis detection
            analysis['crisis'] = self.detect_crisis(text)
            
            # Therapeutic progress analysis
            analysis['progress'] = self.analyze_therapeutic_progress(text)
            
            # Calculate overall sentiment score and label
            sentiment_scores = [
                vader_scores['compound'],
                vader_alt_scores['compound'],
                blob.sentiment.polarity
            ]
            
            if self.transformer_sentiment and 'transformer_sentiment' in analysis:
                # Convert transformer sentiment to numerical score
                trans_score = analysis['transformer_sentiment']['score']
                if analysis['transformer_sentiment']['label'] == 'NEGATIVE':
                    trans_score = -trans_score
                sentiment_scores.append(trans_score)
            
            # Weighted average of sentiment scores
            analysis['sentiment_score'] = np.mean(sentiment_scores)
            
            # Determine sentiment label
            if analysis['sentiment_score'] >= 0.05:
                analysis['sentiment_label'] = 'Positive'
            elif analysis['sentiment_score'] <= -0.05:
                analysis['sentiment_label'] = 'Negative'
            else:
                analysis['sentiment_label'] = 'Neutral'
            
            # Calculate confidence score
            sentiment_variance = np.var(sentiment_scores)
            analysis['confidence'] = max(0.0, 1.0 - sentiment_variance)
            
            # Add analysis metadata
            analysis['analysis_duration'] = (datetime.now() - analysis_start_time).total_seconds()
            analysis['analysis_version'] = '1.0'
            analysis['modality'] = 'text'
            
            # Store in history for personalization
            self.analysis_history.append(analysis)
            
            # Update user patterns if user_id provided
            if user_id:
                self.update_user_patterns(user_id, analysis)
            
            self.logger.info(f"Text analysis completed - Sentiment: {analysis['sentiment_label']}, Crisis Risk: {analysis['crisis']['risk_level']}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return self.create_error_analysis(str(e))
    
    def create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis result for invalid input"""
        return {
            'timestamp': datetime.now(),
            'original_text': '',
            'sentiment_score': 0.0,
            'sentiment_label': 'Neutral',
            'confidence': 0.0,
            'crisis': {'crisis_risk': 0.0, 'risk_level': 'low'},
            'emotions': {'dominant_emotion': 'neutral'},
            'error': 'Empty or invalid text input'
        }
    
    def create_error_analysis(self, error_message: str) -> Dict[str, Any]:
        """Create error analysis result"""
        return {
            'timestamp': datetime.now(),
            'sentiment_score': 0.0,
            'sentiment_label': 'Neutral',
            'confidence': 0.0,
            'crisis': {'crisis_risk': 0.0, 'risk_level': 'low'},
            'emotions': {'dominant_emotion': 'neutral'},
            'error': error_message
        }
    
    def update_user_patterns(self, user_id: str, analysis: Dict[str, Any]):
        """Update user-specific patterns for personalization"""
        
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                'sentiment_history': [],
                'emotion_patterns': {},
                'crisis_history': [],
                'common_words': {},
                'session_count': 0,
                'first_seen': datetime.now()
            }
        
        user_data = self.user_patterns[user_id]
        
        # Update sentiment history
        user_data['sentiment_history'].append({
            'timestamp': analysis['timestamp'],
            'score': analysis['sentiment_score'],
            'label': analysis['sentiment_label']
        })
        
        # Keep only last 100 sentiment records
        if len(user_data['sentiment_history']) > 100:
            user_data['sentiment_history'] = user_data['sentiment_history'][-100:]
        
        # Update emotion patterns
        dominant_emotion = analysis['emotions']['dominant_emotion']
        if dominant_emotion in user_data['emotion_patterns']:
            user_data['emotion_patterns'][dominant_emotion] += 1
        else:
            user_data['emotion_patterns'][dominant_emotion] = 1
        
        # Update crisis history if risk detected
        if analysis['crisis']['crisis_risk'] > 0.3:
            user_data['crisis_history'].append({
                'timestamp': analysis['timestamp'],
                'risk_level': analysis['crisis']['risk_level'],
                'risk_score': analysis['crisis']['crisis_risk']
            })
        
        # Update common words (for personalization)
        words = analysis['processed_text'].split()
        for word in words:
            if word not in self.stop_words and len(word) > 3:
                if word in user_data['common_words']:
                    user_data['common_words'][word] += 1
                else:
                    user_data['common_words'][word] = 1
        
        user_data['session_count'] += 1
        user_data['last_seen'] = datetime.now()
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate insights about user patterns"""
        
        if user_id not in self.user_patterns:
            return {'error': 'No data available for user'}
        
        user_data = self.user_patterns[user_id]
        insights = {}
        
        # Sentiment trends
        if user_data['sentiment_history']:
            recent_sentiments = [s['score'] for s in user_data['sentiment_history'][-10:]]
            insights['recent_sentiment_trend'] = np.mean(recent_sentiments)
            insights['sentiment_stability'] = 1.0 - np.std(recent_sentiments) if len(recent_sentiments) > 1 else 1.0
            
            # Calculate improvement/decline
            if len(user_data['sentiment_history']) >= 5:
                early_avg = np.mean([s['score'] for s in user_data['sentiment_history'][:5]])
                late_avg = np.mean([s['score'] for s in user_data['sentiment_history'][-5:]])
                insights['sentiment_change'] = late_avg - early_avg
        
        # Emotion patterns
        if user_data['emotion_patterns']:
            total_emotions = sum(user_data['emotion_patterns'].values())
            insights['emotion_distribution'] = {
                emotion: count / total_emotions 
                for emotion, count in user_data['emotion_patterns'].items()
            }
            insights['dominant_emotions'] = sorted(
                user_data['emotion_patterns'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
        
        # Crisis risk assessment
        if user_data['crisis_history']:
            recent_crises = [c for c in user_data['crisis_history'] 
                           if c['timestamp'] > datetime.now() - timedelta(days=30)]
            insights['recent_crisis_incidents'] = len(recent_crises)
            insights['crisis_trend'] = 'increasing' if len(recent_crises) > len(user_data['crisis_history']) / 2 else 'stable'
        else:
            insights['recent_crisis_incidents'] = 0
            insights['crisis_trend'] = 'stable'
        
        # Common themes
        if user_data['common_words']:
            insights['common_themes'] = sorted(
                user_data['common_words'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        
        # Session statistics
        insights['total_sessions'] = user_data['session_count']
        insights['days_active'] = (datetime.now() - user_data['first_seen']).days
        insights['engagement_frequency'] = user_data['session_count'] / max(insights['days_active'], 1)
        
        return insights
    
    def batch_analyze(self, texts: List[str], user_id: str = None) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch"""
        
        results = []
        for text in texts:
            try:
                result = self.analyze(text, user_id)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch analysis failed for text: {e}")
                results.append(self.create_error_analysis(str(e)))
        
        return results
    
    def export_analysis_history(self, filepath: str = None) -> str:
        """Export analysis history to JSON file"""
        
        if not filepath:
            filepath = f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Prepare data for export (convert datetime objects)
            export_data = []
            for analysis in self.analysis_history:
                export_analysis = analysis.copy()
                if 'timestamp' in export_analysis:
                    export_analysis['timestamp'] = export_analysis['timestamp'].isoformat()
                export_data.append(export_analysis)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Analysis history exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
    
    def import_analysis_history(self, filepath: str):
        """Import analysis history from JSON file"""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Convert timestamp strings back to datetime objects
            for analysis in import_data:
                if 'timestamp' in analysis and isinstance(analysis['timestamp'], str):
                    analysis['timestamp'] = datetime.fromisoformat(analysis['timestamp'])
            
            self.analysis_history.extend(import_data)
            self.logger.info(f"Imported {len(import_data)} analysis records from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            raise
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about analysis history"""
        
        if not self.analysis_history:
            return {'error': 'No analysis history available'}
        
        stats = {
            'total_analyses': len(self.analysis_history),
            'date_range': {
                'first': min(a['timestamp'] for a in self.analysis_history),
                'last': max(a['timestamp'] for a in self.analysis_history)
            },
            'sentiment_distribution': {'Positive': 0, 'Negative': 0, 'Neutral': 0},
            'average_sentiment_score': 0.0,
            'crisis_incidents': 0,
            'high_risk_incidents': 0,
            'emotion_frequency': {},
            'word_count_stats': {},
            'analysis_performance': {}
        }
        
        # Calculate statistics
        sentiment_scores = []
        word_counts = []
        analysis_durations = []
        
        for analysis in self.analysis_history:
            # Sentiment distribution
            sentiment_label = analysis.get('sentiment_label', 'Neutral')
            stats['sentiment_distribution'][sentiment_label] += 1
            
            # Sentiment scores
            if 'sentiment_score' in analysis:
                sentiment_scores.append(analysis['sentiment_score'])
            
            # Crisis incidents
            if 'crisis' in analysis:
                crisis_risk = analysis['crisis'].get('crisis_risk', 0)
                if crisis_risk > 0.3:
                    stats['crisis_incidents'] += 1
                if crisis_risk > 0.7:
                    stats['high_risk_incidents'] += 1
            
            # Emotion tracking
            if 'emotions' in analysis:
                emotion = analysis['emotions'].get('dominant_emotion', 'neutral')
                stats['emotion_frequency'][emotion] = stats['emotion_frequency'].get(emotion, 0) + 1
            
            # Performance metrics
            if 'word_count' in analysis:
                word_counts.append(analysis['word_count'])
            
            if 'analysis_duration' in analysis:
                analysis_durations.append(analysis['analysis_duration'])
        
        # Calculate averages and statistics
        if sentiment_scores:
            stats['average_sentiment_score'] = np.mean(sentiment_scores)
            stats['sentiment_std'] = np.std(sentiment_scores)
        
        if word_counts:
            stats['word_count_stats'] = {
                'average': np.mean(word_counts),
                'median': np.median(word_counts),
                'min': min(word_counts),
                'max': max(word_counts)
            }
        
        if analysis_durations:
            stats['analysis_performance'] = {
                'average_duration': np.mean(analysis_durations),
                'median_duration': np.median(analysis_durations),
                'total_processing_time': sum(analysis_durations)
            }
        
        # Calculate percentages
        total = stats['total_analyses']
        for sentiment in stats['sentiment_distribution']:
            count = stats['sentiment_distribution'][sentiment]
            stats['sentiment_distribution'][sentiment] = {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            }
        
        stats['crisis_rate'] = (stats['crisis_incidents'] / total * 100) if total > 0 else 0
        stats['high_risk_rate'] = (stats['high_risk_incidents'] / total * 100) if total > 0 else 0
        
        return stats
    
    def clear_history(self, user_id: str = None):
        """Clear analysis history (optionally for specific user)"""
        
        if user_id:
            # Clear user-specific data
            if user_id in self.user_patterns:
                del self.user_patterns[user_id]
                self.logger.info(f"Cleared history for user: {user_id}")
        else:
            # Clear all history
            self.analysis_history = []
            self.user_patterns = {}
            self.logger.info("Cleared all analysis history")
    
    def save_models(self):
        """Save trained models to disk"""
        
        try:
            if hasattr(self, 'emotion_classifier') and self.emotion_classifier:
                joblib.dump(self.emotion_classifier, self.emotion_model_path)
            
            if hasattr(self, 'crisis_detector') and self.crisis_detector:
                joblib.dump(self.crisis_detector, self.crisis_model_path)
            
            if hasattr(self, 'vectorizer') and self.vectorizer:
                joblib.dump(self.vectorizer, self.vectorizer_path)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise
    
    def update_crisis_keywords(self, new_keywords: Dict[str, Dict[str, List[str]]]):
        """Update crisis detection keywords"""
        
        self.crisis_keywords.update(new_keywords)
        self.logger.info("Crisis keywords updated")
        
        # Retrain crisis detector with new keywords
        self.train_crisis_detector()
    
    def validate_analysis_quality(self, text: str, expected_sentiment: str = None) -> Dict[str, Any]:
        """Validate analysis quality against expected results"""
        
        analysis = self.analyze(text)
        
        validation = {
            'text': text,
            'predicted_sentiment': analysis['sentiment_label'],
            'confidence': analysis['confidence'],
            'analysis_consistent': True,
            'warnings': []
        }
        
        if expected_sentiment:
            validation['expected_sentiment'] = expected_sentiment
            validation['prediction_correct'] = analysis['sentiment_label'].lower() == expected_sentiment.lower()
        
        # Check for inconsistencies between different analyzers
        vader_label = 'Positive' if analysis['vader_sentiment']['compound'] > 0.05 else 'Negative' if analysis['vader_sentiment']['compound'] < -0.05 else 'Neutral'
        textblob_label = 'Positive' if analysis['textblob_sentiment']['polarity'] > 0.1 else 'Negative' if analysis['textblob_sentiment']['polarity'] < -0.1 else 'Neutral'
        
        if vader_label != textblob_label:
            validation['analysis_consistent'] = False
            validation['warnings'].append("VADER and TextBlob sentiment analysis disagree")
        
        # Check confidence levels
        if analysis['confidence'] < 0.5:
            validation['warnings'].append("Low confidence analysis")
        
        return validation


# Example usage and testing functions
def main():
    """Example usage of SentimentAnalyzer"""
    
    # Initialize analyzer
    print("Initializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    
    # Test texts for different scenarios
    test_texts = [
        "I'm feeling really great today! Life is wonderful.",
        "I'm so depressed, I don't know what to do anymore.",
        "I've been having thoughts about ending my life.",
        "I'm learning to cope better with my anxiety through therapy.",
        "I feel neutral about everything right now.",
        "I want to hurt myself because I deserve the pain.",
        "I'm grateful for my therapist's help and support."
    ]
    
    print("\n=== Sentiment Analysis Results ===")
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        result = analyzer.analyze(text)
        
        print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.3f})")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Crisis Risk: {result['crisis']['risk_level']} ({result['crisis']['crisis_risk']:.3f})")
        print(f"Dominant Emotion: {result['emotions']['dominant_emotion']}")
        
        if result['crisis']['crisis_risk'] > 0.5:
            print("⚠️  CRISIS ALERT - Immediate attention needed")
    
    # Display statistics
    print("\n=== Analysis Statistics ===")
    stats = analyzer.get_analysis_statistics()
    print(f"Total Analyses: {stats['total_analyses']}")
    print(f"Crisis Rate: {stats['crisis_rate']:.1f}%")
    print(f"Average Sentiment: {stats['average_sentiment_score']:.3f}")
    
    # Export results
    export_file = analyzer.export_analysis_history()
    print(f"\nResults exported to: {export_file}")


if __name__ == "__main__":
    main()