#!/usr/bin/env python3
"""
Setup script for the Multimodal AI Therapy System
Handles initial setup, model downloads, and environment configuration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

class TherapySystemSetup:
    """Setup manager for the AI therapy system"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.setup_logging()
        
        # Required directories
        self.required_dirs = [
            'logs', 'session_data', 'audio_data', 'video_data',
            'multimodal_profiles', 'keys', 'privacy_records', 'models'
        ]
        
        # Core dependencies
        self.core_packages = [
            'streamlit>=1.28.0',
            'pandas>=1.5.0',
            'numpy>=1.24.0',
            'torch>=2.0.0',
            'transformers>=4.30.0'
        ]
        
        # Optional packages by feature
        self.feature_packages = {
            'audio': [
                'librosa>=0.10.0',
                'sounddevice>=0.4.6',
                'soundfile>=0.12.1',
                'speechrecognition>=3.10.0',
                'openai-whisper>=20230314'
            ],
            'video': [
                'opencv-python>=4.8.0',
                'mediapipe>=0.10.0',
                'pillow>=9.5.0'
            ],
            'privacy': [
                'cryptography>=41.0.0'
            ],
            'llm': [
                'groq>=0.4.1',
                'ollama>=0.1.7'
            ]
        }
        
    def setup_logging(self):
        """Setup logging for setup process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.logger.info(f"✓ Python {version.major}.{version.minor} is compatible")
            return True
        else:
            self.logger.error(f"✗ Python {version.major}.{version.minor} is not supported. Requires Python 3.8+")
            return False
    
    def create_directories(self):
        """Create required directory structure"""
        self.logger.info("Creating directory structure...")
        
        for dir_name in self.required_dirs:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True, mode=0o700)
            self.logger.info(f"✓ Created directory: {dir_name}")
    
    def install_packages(self, packages: List[str], feature_name: str = "core"):
        """Install Python packages"""
        self.logger.info(f"Installing {feature_name} packages...")
        
        for package in packages:
            try:
                self.logger.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package, '--upgrade'
                ], stdout=subprocess.DEVNULL)
                self.logger.info(f"✓ Installed {package}")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"⚠ Failed to install {package}: {e}")
                return False
        
        return True
    
    def setup_environment_variables(self):
        """Setup environment variables and configuration"""
        self.logger.info("Setting up environment configuration...")
        
        env_template = {
            'GROQ_API_KEY': 'your_groq_api_key_here',
            'PRIVACY_LEVEL': 'standard',
            'MODEL_CACHE_DIR': str(self.base_dir / 'models'),
            'LOG_LEVEL': 'INFO'
        }
        
        env_file = self.base_dir / '.env'
        
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write("# AI Therapy System Environment Configuration\n")
                f.write("# Copy this file and update with your actual values\n\n")
                for key, value in env_template.items():
                    f.write(f"{key}={value}\n")
            
            self.logger.info(f"✓ Created environment template: {env_file}")
            self.logger.info("  Please update .env file with your API keys")
        else:
            self.logger.info("✓ Environment file already exists")
    
    def download_models(self):
        """Download and cache required models"""
        self.logger.info("Setting up AI models...")
        
        models_to_download = [
            {
                'name': 'Mental Health Analysis',
                'model_id': 'rabiaqayyum/autotrain-mental-health-analysis',
                'type': 'huggingface'
            },
            {
                'name': 'Emotion Recognition',
                'model_id': 'SamLowe/roberta-base-go_emotions',
                'type': 'huggingface'
            },
            {
                'name': 'Speech Emotion Recognition',
                'model_id': 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
                'type': 'huggingface'
            }
        ]
        
        try:
            from transformers import AutoTokenizer, AutoModel, pipeline
            
            for model_info in models_to_download:
                try:
                    self.logger.info(f"Downloading {model_info['name']}...")
                    
                    if 'emotion' in model_info['model_id'].lower():
                        # For emotion models, use pipeline to cache
                        pipeline("text-classification", model=model_info['model_id'])
                    else:
                        # For other models, download tokenizer and model
                        AutoTokenizer.from_pretrained(model_info['model_id'])
                        AutoModel.from_pretrained(model_info['model_id'])
                    
                    self.logger.info(f"✓ Downloaded {model_info['name']}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠ Failed to download {model_info['name']}: {e}")
        
        except ImportError:
            self.logger.warning("⚠ Transformers not available, skipping model downloads")
    
    def setup_ollama(self):
        """Setup local Ollama if available"""
        self.logger.info("Checking for Ollama installation...")
        
        try:
            result = subprocess.run(['ollama', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.logger.info("✓ Ollama is installed")
                
                # Try to pull a model
                self.logger.info("Downloading Llama model...")
                try:
                    subprocess.run(['ollama', 'pull', 'llama3.1:8b'], 
                                 timeout=300, check=True)
                    self.logger.info("✓ Downloaded Llama 3.1 8B model")
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    self.logger.warning("⚠ Failed to download Ollama model (timeout or error)")
            else:
                self.logger.info("ℹ Ollama not found - install from https://ollama.ai for local models")
        
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.info("ℹ Ollama not found - install from https://ollama.ai for local models")
    
    def create_config_file(self):
        """Create system configuration file"""
        self.logger.info("Creating system configuration...")
        
        config = {
            'system': {
                'version': '2.0',
                'setup_date': str(Path.cwd()),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
            },
            'features': {
                'text_analysis': True,
                'audio_analysis': True,
                'visual_analysis': True,
                'multimodal_fusion': True,
                'privacy_protection': True
            },
            'models': {
                'text_emotion_model': 'SamLowe/roberta-base-go_emotions',
                'mental_health_model': 'rabiaqayyum/autotrain-mental-health-analysis',
                'audio_emotion_model': 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
                'llm_provider': 'groq',  # or 'ollama'
                'llm_model': 'llama-3.3-70b-versatile'
            },
            'privacy': {
                'default_level': 'standard',
                'retention_days': 7,
                'encryption_enabled': True,
                'anonymization_enabled': True
            }
        }
        
        config_file = self.base_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"✓ Created configuration file: {config_file}")
    
    def run_tests(self):
        """Run basic system tests"""
        self.logger.info("Running system tests...")
        
        test_results = {
            'imports': self.test_imports(),
            'directories': self.test_directories(),
            'models': self.test_models()
        }
        
        all_passed = all(test_results.values())
        
        if all_passed:
            self.logger.info("✓ All tests passed!")
        else:
            self.logger.warning("⚠ Some tests failed - check logs above")
        
        return all_passed
    
    def test_imports(self) -> bool:
        """Test if core modules can be imported"""
        required_imports = [
            'streamlit',
            'pandas',
            'numpy',
            'torch',
            'transformers'
        ]
        
        for module in required_imports:
            try:
                __import__(module)
                self.logger.info(f"✓ Import test passed: {module}")
            except ImportError:
                self.logger.error(f"✗ Import test failed: {module}")
                return False
        
        return True
    
    def test_directories(self) -> bool:
        """Test if directories exist and are writable"""
        for dir_name in self.required_dirs:
            dir_path = self.base_dir / dir_name
            
            if not dir_path.exists():
                self.logger.error(f"✗ Directory missing: {dir_name}")
                return False
            
            # Test write permissions
            test_file = dir_path / '.test_write'
            try:
                test_file.write_text('test')
                test_file.unlink()
                self.logger.info(f"✓ Directory test passed: {dir_name}")
            except Exception:
                self.logger.error(f"✗ Directory not writable: {dir_name}")
                return False
        
        return True
    
    def test_models(self) -> bool:
        """Test if models can be loaded"""
        try:
            from transformers import pipeline
            
            # Test a simple model
            classifier = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            result = classifier("This is a test")
            self.logger.info("✓ Model loading test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Model loading test failed: {e}")
            return False
    
    def full_setup(self, features: List[str] = None):
        """Run complete setup process"""
        self.logger.info("=" * 60)
        self.logger.info("🧠 AI Therapy System Setup")
        self.logger.info("=" * 60)
        
        if features is None:
            features = ['audio', 'video', 'privacy', 'llm']
        
        setup_steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing core packages", lambda: self.install_packages(self.core_packages, "core")),
        ]
        
        # Add feature-specific installations
        for feature in features:
            if feature in self.feature_packages:
                step_name = f"Installing {feature} packages"
                step_func = lambda f=feature: self.install_packages(self.feature_packages[f], f)
                setup_steps.append((step_name, step_func))
        
        # Add remaining setup steps
        setup_steps.extend([
            ("Setting up environment", self.setup_environment_variables),
            ("Downloading AI models", self.download_models),
            ("Setting up Ollama (optional)", self.setup_ollama),
            ("Creating configuration", self.create_config_file),
            ("Running tests", self.run_tests)
        ])
        
        # Execute setup steps
        failed_steps = []
        
        for step_name, step_func in setup_steps:
            self.logger.info(f"\n🔄 {step_name}...")
            try:
                success = step_func()
                if success is False:
                    failed_steps.append(step_name)
                    self.logger.error(f"✗ Failed: {step_name}")
                else:
                    self.logger.info(f"✓ Completed: {step_name}")
            except Exception as e:
                failed_steps.append(step_name)
                self.logger.error(f"✗ Error in {step_name}: {e}")
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        if not failed_steps:
            self.logger.info("🎉 Setup completed successfully!")
            self.logger.info("\n📋 Next steps:")
            self.logger.info("1. Update .env file with your API keys")
            self.logger.info("2. Run: streamlit run app.py")
            self.logger.info("3. Open http://localhost:8501 in your browser")
        else:
            self.logger.warning(f"⚠ Setup completed with {len(failed_steps)} issues:")
            for step in failed_steps:
                self.logger.warning(f"  - {step}")
        
        self.logger.info("=" * 60)
    
    def interactive_setup(self):
        """Interactive setup with user choices"""
        self.logger.info("🧠 Welcome to AI Therapy System Interactive Setup\n")
        
        # Feature selection
        self.logger.info("Select features to install:")
        features = []
        
        feature_options = [
            ("audio", "Audio analysis (speech emotion, prosody)"),
            ("video", "Visual analysis (facial emotions, body language)"),
            ("privacy", "Privacy protection and encryption"),
            ("llm", "Large Language Model integration (Groq/Ollama)")
        ]
        
        for feature_key, description in feature_options:
            response = input(f"Install {description}? (y/n) [y]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                features.append(feature_key)
                self.logger.info(f"✓ Selected: {feature_key}")
        
        # LLM provider selection
        if 'llm' in features:
            self.logger.info("\nSelect LLM provider:")
            self.logger.info("1. Groq (cloud-based, fast, requires API key)")
            self.logger.info("2. Ollama (local, private, slower)")
            self.logger.info("3. Both")
            
            llm_choice = input("Choice (1/2/3) [1]: ").strip()
            if llm_choice == '2':
                self.logger.info("Selected: Ollama only")
            elif llm_choice == '3':
                self.logger.info("Selected: Both Groq and Ollama")
            else:
                self.logger.info("Selected: Groq")
        
        # Privacy level
        if 'privacy' in features:
            self.logger.info("\nSelect default privacy level:")
            self.logger.info("1. Minimal (no storage, immediate anonymization)")
            self.logger.info("2. Standard (limited storage, 7-day retention)")
            self.logger.info("3. Full Session (complete data, 30-day retention)")
            
            privacy_choice = input("Choice (1/2/3) [2]: ").strip()
            privacy_levels = {'1': 'minimal', '2': 'standard', '3': 'full_session'}
            privacy_level = privacy_levels.get(privacy_choice, 'standard')
            self.logger.info(f"Selected: {privacy_level} privacy level")
        
        # Confirm and proceed
        self.logger.info(f"\nProceeding with setup for features: {', '.join(features)}")
        proceed = input("Continue? (y/n) [y]: ").strip().lower()
        
        if proceed in ['', 'y', 'yes']:
            self.full_setup(features)
        else:
            self.logger.info("Setup cancelled.")

def main():
    """Main setup entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Therapy System Setup")
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run interactive setup')
    parser.add_argument('--features', nargs='+', 
                       choices=['audio', 'video', 'privacy', 'llm'],
                       help='Features to install')
    parser.add_argument('--minimal', action='store_true',
                       help='Minimal installation (core only)')
    parser.add_argument('--test-only', action='store_true',
                       help='Run tests only')
    
    args = parser.parse_args()
    
    setup = TherapySystemSetup()
    
    if args.test_only:
        setup.run_tests()
    elif args.interactive:
        setup.interactive_setup()
    elif args.minimal:
        setup.full_setup(features=[])
    else:
        features = args.features if args.features else ['audio', 'video', 'privacy', 'llm']
        setup.full_setup(features)

if __name__ == "__main__":
    main()

# Additional utility functions for post-setup
class SystemManager:
    """Manage the therapy system after setup"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.config_file = self.base_dir / 'config.json'
    
    def load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def update_config(self, updates: Dict[str, Any]):
        """Update system configuration"""
        config = self.load_config()
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        config = deep_update(config, updates)
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        config = self.load_config()
        
        status = {
            'setup_complete': self.config_file.exists(),
            'version': config.get('system', {}).get('version', 'unknown'),
            'features_enabled': config.get('features', {}),
            'models_configured': config.get('models', {}),
            'directories_exist': all(
                (self.base_dir / d).exists() 
                for d in ['logs', 'session_data', 'models']
            )
        }
        
        # Check if models are actually available
        try:
            import transformers
            status['transformers_available'] = True
        except ImportError:
            status['transformers_available'] = False
        
        # Check Ollama
        try:
            result = subprocess.run(['ollama', 'version'], 
                                  capture_output=True, timeout=5)
            status['ollama_available'] = result.returncode == 0
        except:
            status['ollama_available'] = False
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health = {
            'overall': 'healthy',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check disk space
        import shutil
        free_space = shutil.disk_usage(self.base_dir).free / (1024**3)  # GB
        
        if free_space < 1:
            health['issues'].append('Low disk space (< 1GB free)')
            health['overall'] = 'unhealthy'
        elif free_space < 5:
            health['warnings'].append('Limited disk space (< 5GB free)')
        
        # Check log files
        logs_dir = self.base_dir / 'logs'
        if logs_dir.exists():
            log_files = list(logs_dir.glob('*.log'))
            total_log_size = sum(f.stat().st_size for f in log_files) / (1024**2)  # MB
            
            if total_log_size > 100:
                health['warnings'].append(f'Large log files ({total_log_size:.1f}MB)')
                health['recommendations'].append('Consider log rotation')
        
        # Check for expired sessions
        session_dir = self.base_dir / 'session_data'
        if session_dir.exists():
            session_files = list(session_dir.glob('*_metadata.json'))
            if len(session_files) > 100:
                health['warnings'].append(f'Many session files ({len(session_files)})')
                health['recommendations'].append('Run cleanup_expired_sessions()')
        
        return health
    
    def cleanup_system(self):
        """Cleanup temporary files and logs"""
        cleanup_summary = {
            'logs_cleaned': 0,
            'temp_files_removed': 0,
            'space_freed_mb': 0
        }
        
        # Clean old logs
        logs_dir = self.base_dir / 'logs'
        if logs_dir.exists():
            import time
            current_time = time.time()
            week_ago = current_time - (7 * 24 * 60 * 60)
            
            for log_file in logs_dir.glob('*.log'):
                if log_file.stat().st_mtime < week_ago:
                    size = log_file.stat().st_size
                    log_file.unlink()
                    cleanup_summary['logs_cleaned'] += 1
                    cleanup_summary['space_freed_mb'] += size / (1024**2)
        
        # Clean temporary files
        temp_patterns = ['*.tmp', '*.temp', '.DS_Store', 'Thumbs.db']
        for pattern in temp_patterns:
            for temp_file in self.base_dir.rglob(pattern):
                size = temp_file.stat().st_size
                temp_file.unlink()
                cleanup_summary['temp_files_removed'] += 1
                cleanup_summary['space_freed_mb'] += size / (1024**2)
        
        return cleanup_summary

# CLI commands for system management
def cli_status():
    """CLI command to check system status"""
    manager = SystemManager()
    status = manager.get_system_status()
    
    print("🧠 AI Therapy System Status")
    print("=" * 40)
    print(f"Setup Complete: {'✓' if status['setup_complete'] else '✗'}")
    print(f"Version: {status['version']}")
    print(f"Transformers: {'✓' if status['transformers_available'] else '✗'}")
    print(f"Ollama: {'✓' if status['ollama_available'] else '✗'}")
    
    print("\nFeatures:")
    for feature, enabled in status['features_enabled'].items():
        print(f"  {feature}: {'✓' if enabled else '✗'}")

def cli_health():
    """CLI command to check system health"""
    manager = SystemManager()
    health = manager.health_check()
    
    print("🏥 System Health Check")
    print("=" * 40)
    print(f"Overall Status: {health['overall'].upper()}")
    
    if health['issues']:
        print("\n❌ Issues:")
        for issue in health['issues']:
            print(f"  - {issue}")
    
    if health['warnings']:
        print("\n⚠️ Warnings:")
        for warning in health['warnings']:
            print(f"  - {warning}")
    
    if health['recommendations']:
        print("\n💡 Recommendations:")
        for rec in health['recommendations']:
            print(f"  - {rec}")

def cli_cleanup():
    """CLI command to cleanup system"""
    manager = SystemManager()
    result = manager.cleanup_system()
    
    print("🧹 System Cleanup Complete")
    print("=" * 40)
    print(f"Logs cleaned: {result['logs_cleaned']}")
    print(f"Temp files removed: {result['temp_files_removed']}")
    print(f"Space freed: {result['space_freed_mb']:.1f} MB")

# Make CLI commands available
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'status':
            cli_status()
        elif command == 'health':
            cli_health()
        elif command == 'cleanup':
            cli_cleanup()
        else:
            main()
    else:
        main()