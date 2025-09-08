"""
Configuration Management for AI Therapy System
Phase 1: Simplified configuration for text-only system
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # HuggingFace models (verified working)
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda
    cache_dir: Optional[str] = None
    
    # Model parameters
    max_length: int = 512
    batch_size: int = 1

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    ollama_timeout: int = 30
    
    # Groq settings
    groq_api_key: Optional[str] = None
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = "mixtral-8x7b-32768"
    groq_timeout: int = 20
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 300
    top_p: float = 0.9

@dataclass
class AnalysisConfig:
    """Configuration for text analysis"""
    # Crisis detection thresholds
    crisis_threshold_high: float = 0.8
    crisis_threshold_moderate: float = 0.5
    crisis_threshold_low: float = 0.2
    
    # Risk assessment weights
    crisis_weight: float = 0.5
    sentiment_weight: float = 0.3
    emotion_weight: float = 0.2
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.3
    high_confidence_threshold: float = 0.7
    
    # Text processing
    min_word_count: int = 1
    max_word_count: int = 1000

@dataclass
class ServerConfig:
    """Configuration for Flask server"""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    secret_key: str = "dev-key-change-in-production"
    
    # Session management
    session_timeout_hours: int = 24
    max_active_sessions: int = 1000
    cleanup_interval_minutes: int = 60

@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file: str = "logs/app.log"
    max_file_size_mb: int = 10
    backup_count: int = 5

@dataclass
class SecurityConfig:
    """Configuration for security settings"""
    # Crisis logging
    log_crisis_events: bool = True
    crisis_log_file: str = "logs/crisis_events.jsonl"
    
    # Data retention
    anonymize_logs: bool = True
    data_retention_days: int = 30
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.models = ModelConfig()
        self.llm = LLMConfig()
        self.analysis = AnalysisConfig()
        self.server = ServerConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Apply configuration
        self._apply_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Server config
        self.server.host = os.getenv('HOST', self.server.host)
        self.server.port = int(os.getenv('PORT', self.server.port))
        self.server.debug = os.getenv('FLASK_ENV') == 'development'
        self.server.secret_key = os.getenv('SECRET_KEY', self.server.secret_key)
        
        # LLM config
        self.llm.ollama_url = os.getenv('OLLAMA_URL', self.llm.ollama_url)
        self.llm.ollama_model = os.getenv('OLLAMA_MODEL', self.llm.ollama_model)
        self.llm.groq_api_key = os.getenv('GROQ_API_KEY', self.llm.groq_api_key)
        self.llm.groq_model = os.getenv('GROQ_MODEL', self.llm.groq_model)
        
        # Model config
        device = os.getenv('DEVICE', 'auto')
        if device != 'auto':
            self.models.device = device
        
        # Logging config
        log_level = os.getenv('LOG_LEVEL', self.logging.level)
        if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            self.logging.level = log_level
        
        logger.info("Configuration loaded from environment variables")
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'models' in config_data:
                self._update_dataclass(self.models, config_data['models'])
            
            if 'llm' in config_data:
                self._update_dataclass(self.llm, config_data['llm'])
            
            if 'analysis' in config_data:
                self._update_dataclass(self.analysis, config_data['analysis'])
            
            if 'server' in config_data:
                self._update_dataclass(self.server, config_data['server'])
            
            if 'logging' in config_data:
                self._update_dataclass(self.logging, config_data['logging'])
            
            if 'security' in config_data:
                self._update_dataclass(self.security, config_data['security'])
            
            logger.info(f"Configuration loaded from file: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass object with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _apply_config(self):
        """Apply configuration settings"""
        
        # Set up logging
        self._setup_logging()
        
        # Create necessary directories
        self._create_directories()
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.logging.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, self.logging.level))
        
        # Configure logging format
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        # File handler (if enabled)
        if self.logging.log_to_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.log_file,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        
        logger.info("Logging configured successfully")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            'logs',
            'data',
            'cache'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def _validate_config(self):
        """Validate configuration settings"""
        
        # Validate thresholds
        if not (0 <= self.analysis.crisis_threshold_low <= 
                self.analysis.crisis_threshold_moderate <= 
                self.analysis.crisis_threshold_high <= 1):
            raise ValueError("Crisis thresholds must be in ascending order between 0 and 1")
        
        # Validate weights
        total_weight = (self.analysis.crisis_weight + 
                       self.analysis.sentiment_weight + 
                       self.analysis.emotion_weight)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Analysis weights sum to {total_weight}, should sum to 1.0")
        
        # Validate ports
        if not (1 <= self.server.port <= 65535):
            raise ValueError(f"Invalid port number: {self.server.port}")
        
        logger.info("Configuration validation completed")
    
    def save_to_file(self, filename: str):
        """Save current configuration to file"""
        config_data = {
            'models': asdict(self.models),
            'llm': asdict(self.llm),
            'analysis': asdict(self.analysis),
            'server': asdict(self.server),
            'logging': asdict(self.logging),
            'security': asdict(self.security)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'models': {
                'emotion_model': self.models.emotion_model,
                'sentiment_model': self.models.sentiment_model,
                'device': self.models.device
            },
            'llm': {
                'ollama_available': self.llm.ollama_url is not None,
                'groq_available': self.llm.groq_api_key is not None,
                'primary_model': self.llm.ollama_model
            },
            'server': {
                'host': self.server.host,
                'port': self.server.port,
                'debug': self.server.debug
            },
            'security': {
                'crisis_logging': self.security.log_crisis_events,
                'data_anonymization': self.security.anonymize_logs
            }
        }
    
    def update_llm_api_key(self, groq_api_key: str):
        """Update Groq API key"""
        self.llm.groq_api_key = groq_api_key
        logger.info("Groq API key updated")
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.server.debug
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.server.debug


# Global configuration instance
config = None

def get_config(config_file: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global config
    if config is None:
        config = Config(config_file)
    return config

def reload_config(config_file: Optional[str] = None):
    """Reload configuration"""
    global config
    config = Config(config_file)
    logger.info("Configuration reloaded")


# Example configuration file content
EXAMPLE_CONFIG = {
    "models": {
        "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
        "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "device": "auto",
        "max_length": 512
    },
    "llm": {
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama2",
        "groq_api_key": "your-groq-api-key-here",
        "temperature": 0.7,
        "max_tokens": 300
    },
    "analysis": {
        "crisis_threshold_high": 0.8,
        "crisis_threshold_moderate": 0.5,
        "crisis_threshold_low": 0.2
    },
    "server": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
        "secret_key": "your-secret-key-here"
    },
    "logging": {
        "level": "INFO",
        "log_to_file": True,
        "log_file": "logs/app.log"
    },
    "security": {
        "log_crisis_events": True,
        "anonymize_logs": True,
        "data_retention_days": 30
    }
}


def create_example_config_file(filename: str = "config.json"):
    """Create an example configuration file"""
    try:
        with open(filename, 'w') as f:
            json.dump(EXAMPLE_CONFIG, f, indent=2)
        print(f"Example configuration file created: {filename}")
        print("Please edit the file to match your environment before use.")
    except Exception as e:
        print(f"Failed to create config file: {e}")


if __name__ == "__main__":
    # Create example config file
    create_example_config_file()
    
    # Test configuration loading
    config = Config()
    print("\nConfiguration Summary:")
    print(json.dumps(config.get_summary(), indent=2))
