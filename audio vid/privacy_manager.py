import hashlib
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class PrivacyManager:
    """
    Privacy Manager for handling data protection, anonymization, and compliance
    in the mental health therapy system
    """
    
    def __init__(self):
        self.data_retention_days = 30  # Default retention period
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Privacy settings
        self.privacy_levels = {
            'minimal': {
                'store_text': False,
                'store_audio': False,
                'store_video': False,
                'anonymize_immediately': True,
                'retention_days': 1
            },
            'standard': {
                'store_text': True,
                'store_audio': False,
                'store_video': False,
                'anonymize_immediately': False,
                'retention_days': 7
            },
            'full_session': {
                'store_text': True,
                'store_audio': True,
                'store_video': True,
                'anonymize_immediately': False,
                'retention_days': 30
            }
        }
        
        # Ensure directories exist
        self.ensure_directories()
        
        # Initialize privacy audit log
        self.audit_logger = self.setup_audit_logging()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get existing encryption key or create new one"""
        key_file = 'keys/encryption.key'
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Create new key
            os.makedirs('keys', exist_ok=True)
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            return key
    
    def ensure_directories(self):
        """Ensure all necessary directories exist with proper permissions"""
        directories = [
            'logs',
            'session_data',
            'audio_data', 
            'video_data',
            'multimodal_profiles',
            'keys',
            'privacy_records'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            # Set restrictive permissions (owner only)
            os.chmod(directory, 0o700)
    
    def setup_audit_logging(self):
        """Setup privacy audit logging"""
        audit_logger = logging.getLogger('privacy_audit')
        audit_logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler('logs/privacy_audit.log')
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not audit_logger.handlers:
            audit_logger.addHandler(handler)
        
        return audit_logger
    
    def generate_session_id(self, include_timestamp: bool = True) -> str:
        """Generate anonymous session ID"""
        base_id = str(uuid.uuid4())
        
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d')
            session_id = f"session_{timestamp}_{base_id[:8]}"
        else:
            session_id = f"session_{base_id[:12]}"
        
        self.audit_logger.info(f"Generated new session ID: {session_id}")
        return session_id
    
    def anonymize_text(self, text: str, session_id: str) -> Dict[str, Any]:
        """Anonymize text data by removing/masking PII"""
        
        original_text = text
        anonymized_text = text
        pii_found = []
        
        # Email addresses
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            pii_found.extend(['email'] * len(emails))
            anonymized_text = re.sub(email_pattern, '[EMAIL_REDACTED]', anonymized_text)
        
        # Phone numbers
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            pii_found.extend(['phone'] * len(phones))
            anonymized_text = re.sub(phone_pattern, '[PHONE_REDACTED]', anonymized_text)
        
        # Social Security Numbers
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        ssns = re.findall(ssn_pattern, text)
        if ssns:
            pii_found.extend(['ssn'] * len(ssns))
            anonymized_text = re.sub(ssn_pattern, '[SSN_REDACTED]', anonymized_text)
        
        # Names (simple pattern - could be enhanced with NER)
        # This is a basic implementation - in production, use proper NER
        common_name_indicators = [
            r'\bmy name is \w+\b',
            r'\bi am \w+ \w+\b',
            r'\bi\'m \w+ \w+\b'
        ]
        
        for pattern in common_name_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                pii_found.append('name')
                anonymized_text = re.sub(pattern, '[NAME_REDACTED]', anonymized_text, flags=re.IGNORECASE)
        
        # Log PII detection
        if pii_found:
            self.audit_logger.warning(
                f"PII detected and anonymized in session {session_id}: {set(pii_found)}"
            )
        
        return {
            'original_length': len(original_text),
            'anonymized_text': anonymized_text,
            'anonymized_length': len(anonymized_text),
            'pii_types_found': list(set(pii_found)),
            'pii_count': len(pii_found),
            'anonymization_timestamp': datetime.now().isoformat()
        }
    
    def encrypt_data(self, data: Any, data_type: str = 'text') -> Dict[str, Any]:
        """Encrypt sensitive data"""
        
        try:
            # Convert data to JSON string if not already string
            if isinstance(data, str):
                data_str = data
            else:
                data_str = json.dumps(data)
            
            # Encrypt the data
            encrypted_data = self.cipher_suite.encrypt(data_str.encode())
            
            # Create metadata
            encryption_metadata = {
                'encrypted_data': base64.b64encode(encrypted_data).decode(),
                'data_type': data_type,
                'encryption_timestamp': datetime.now().isoformat(),
                'encryption_method': 'Fernet',
                'data_hash': hashlib.sha256(data_str.encode()).hexdigest()[:16]  # First 16 chars for verification
            }
            
            self.audit_logger.info(f"Data encrypted: type={data_type}, hash={encryption_metadata['data_hash']}")
            
            return encryption_metadata
            
        except Exception as e:
            self.audit_logger.error(f"Encryption failed: {e}")
            raise Exception(f"Data encryption failed: {e}")
    
    def decrypt_data(self, encrypted_metadata: Dict[str, Any]) -> Any:
        """Decrypt previously encrypted data"""
        
        try:
            # Extract encrypted data
            encrypted_data = base64.b64decode(encrypted_metadata['encrypted_data'].encode())
            
            # Decrypt
            decrypted_bytes = self.cipher_suite.decrypt(encrypted_data)
            decrypted_str = decrypted_bytes.decode()
            
            # Verify hash
            data_hash = hashlib.sha256(decrypted_str.encode()).hexdigest()[:16]
            if data_hash != encrypted_metadata.get('data_hash', ''):
                self.audit_logger.warning("Data hash mismatch during decryption")
            
            # Convert back to original format
            data_type = encrypted_metadata.get('data_type', 'text')
            if data_type == 'text':
                return decrypted_str
            else:
                return json.loads(decrypted_str)
                
        except Exception as e:
            self.audit_logger.error(f"Decryption failed: {e}")
            raise Exception(f"Data decryption failed: {e}")
    
    def store_session_data(self, session_id: str, data: Dict[str, Any], 
                          privacy_level: str = 'standard') -> Dict[str, Any]:
        """Store session data according to privacy settings"""
        
        settings = self.privacy_levels.get(privacy_level, self.privacy_levels['standard'])
        storage_result = {
            'session_id': session_id,
            'privacy_level': privacy_level,
            'storage_timestamp': datetime.now().isoformat(),
            'stored_components': [],
            'anonymized_components': [],
            'retention_until': (datetime.now() + timedelta(days=settings['retention_days'])).isoformat()
        }
        
        # Process text data
        if 'text' in data and settings['store_text']:
            text_data = data['text']
            
            if settings['anonymize_immediately']:
                anonymized = self.anonymize_text(text_data, session_id)
                text_to_store = anonymized['anonymized_text']
                storage_result['anonymized_components'].append('text')
            else:
                text_to_store = text_data
            
            # Encrypt and store
            encrypted_text = self.encrypt_data(text_to_store, 'text')
            self._save_encrypted_data(session_id, 'text', encrypted_text)
            storage_result['stored_components'].append('text')
        
        # Process audio data
        if 'audio' in data and settings['store_audio']:
            # For audio, we typically store analysis results, not raw audio
            audio_analysis = data['audio']
            encrypted_audio = self.encrypt_data(audio_analysis, 'audio_analysis')
            self._save_encrypted_data(session_id, 'audio_analysis', encrypted_audio)
            storage_result['stored_components'].append('audio_analysis')
        
        # Process video data
        if 'video' in data and settings['store_video']:
            # Similar to audio, store analysis results
            video_analysis = data['video']
            encrypted_video = self.encrypt_data(video_analysis, 'video_analysis')
            self._save_encrypted_data(session_id, 'video_analysis', encrypted_video)
            storage_result['stored_components'].append('video_analysis')
        
        # Store session metadata
        session_metadata = {
            'session_id': session_id,
            'privacy_level': privacy_level,
            'creation_time': storage_result['storage_timestamp'],
            'retention_until': storage_result['retention_until'],
            'components': storage_result['stored_components']
        }
        
        self._save_session_metadata(session_id, session_metadata)
        
        # Log the storage operation
        self.audit_logger.info(
            f"Session data stored: {session_id}, privacy_level={privacy_level}, "
            f"components={storage_result['stored_components']}"
        )
        
        return storage_result
    
    def _save_encrypted_data(self, session_id: str, data_type: str, encrypted_data: Dict[str, Any]):
        """Save encrypted data to file"""
        
        filepath = f"session_data/{session_id}_{data_type}.json"
        with open(filepath, 'w') as f:
            json.dump(encrypted_data, f, indent=2)
        
        # Set restrictive permissions
        os.chmod(filepath, 0o600)
    
    def _save_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Save session metadata"""
        
        filepath = f"session_data/{session_id}_metadata.json"
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        os.chmod(filepath, 0o600)
    
    def retrieve_session_data(self, session_id: str, data_type: str = None) -> Dict[str, Any]:
        """Retrieve and decrypt session data"""
        
        try:
            # Check if session exists and is not expired
            if not self._is_session_valid(session_id):
                raise Exception(f"Session {session_id} not found or expired")
            
            if data_type:
                # Retrieve specific data type
                filepath = f"session_data/{session_id}_{data_type}.json"
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        encrypted_data = json.load(f)
                    
                    decrypted_data = self.decrypt_data(encrypted_data)
                    
                    self.audit_logger.info(f"Data retrieved: session={session_id}, type={data_type}")
                    return {data_type: decrypted_data}
                else:
                    return {}
            else:
                # Retrieve all available data for session
                session_data = {}
                metadata = self._load_session_metadata(session_id)
                
                for component in metadata.get('components', []):
                    component_data = self.retrieve_session_data(session_id, component)
                    session_data.update(component_data)
                
                return session_data
                
        except Exception as e:
            self.audit_logger.error(f"Data retrieval failed: session={session_id}, error={e}")
            raise
    
    def _is_session_valid(self, session_id: str) -> bool:
        """Check if session exists and is not expired"""
        
        metadata_file = f"session_data/{session_id}_metadata.json"
        if not os.path.exists(metadata_file):
            return False
        
        try:
            metadata = self._load_session_metadata(session_id)
            retention_until = datetime.fromisoformat(metadata['retention_until'])
            
            return datetime.now() < retention_until
            
        except Exception:
            return False
    
    def _load_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Load session metadata"""
        
        filepath = f"session_data/{session_id}_metadata.json"
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def delete_session_data(self, session_id: str, reason: str = "user_request") -> Dict[str, Any]:
        """Securely delete session data"""
        
        deletion_result = {
            'session_id': session_id,
            'deletion_timestamp': datetime.now().isoformat(),
            'deletion_reason': reason,
            'files_deleted': []
        }
        
        try:
            # Get session metadata to know what files to delete
            if self._is_session_valid(session_id):
                metadata = self._load_session_metadata(session_id)
                components = metadata.get('components', [])
                
                # Delete data files
                for component in components:
                    filepath = f"session_data/{session_id}_{component}.json"
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        deletion_result['files_deleted'].append(f"{session_id}_{component}.json")
                
                # Delete metadata file
                metadata_file = f"session_data/{session_id}_metadata.json"
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                    deletion_result['files_deleted'].append(f"{session_id}_metadata.json")
                
                self.audit_logger.info(
                    f"Session data deleted: {session_id}, reason={reason}, "
                    f"files={len(deletion_result['files_deleted'])}"
                )
            
            else:
                self.audit_logger.warning(f"Attempted to delete non-existent session: {session_id}")
                deletion_result['error'] = 'Session not found or already expired'
        
        except Exception as e:
            self.audit_logger.error(f"Session deletion failed: {session_id}, error={e}")
            deletion_result['error'] = str(e)
        
        return deletion_result
    
    def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """Clean up expired session data"""
        
        cleanup_result = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'sessions_deleted': [],
            'files_deleted': 0,
            'errors': []
        }
        
        try:
            # Get all metadata files
            session_dir = 'session_data'
            if not os.path.exists(session_dir):
                return cleanup_result
            
            metadata_files = [f for f in os.listdir(session_dir) if f.endswith('_metadata.json')]
            
            for metadata_file in metadata_files:
                try:
                    session_id = metadata_file.replace('_metadata.json', '')
                    
                    if not self._is_session_valid(session_id):
                        # Session is expired, delete it
                        deletion_result = self.delete_session_data(session_id, "automatic_cleanup")
                        
                        if 'error' not in deletion_result:
                            cleanup_result['sessions_deleted'].append(session_id)
                            cleanup_result['files_deleted'] += len(deletion_result.get('files_deleted', []))
                        else:
                            cleanup_result['errors'].append(f"{session_id}: {deletion_result['error']}")
                
                except Exception as e:
                    cleanup_result['errors'].append(f"{metadata_file}: {str(e)}")
            
            self.audit_logger.info(
                f"Cleanup completed: {len(cleanup_result['sessions_deleted'])} sessions deleted, "
                f"{cleanup_result['files_deleted']} files removed"
            )
        
        except Exception as e:
            cleanup_result['errors'].append(f"Cleanup failed: {str(e)}")
            self.audit_logger.error(f"Cleanup failed: {e}")
        
        return cleanup_result
    
    def get_privacy_report(self, session_id: str = None) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'report_type': 'session_specific' if session_id else 'system_wide',
            'privacy_compliance': {
                'encryption_enabled': True,
                'data_minimization': True,
                'retention_limits': True,
                'anonymization_available': True
            }
        }
        
        if session_id:
            # Session-specific report
            if self._is_session_valid(session_id):
                metadata = self._load_session_metadata(session_id)
                
                report['session_info'] = {
                    'session_id': session_id,
                    'creation_time': metadata['creation_time'],
                    'retention_until': metadata['retention_until'],
                    'privacy_level': metadata['privacy_level'],
                    'stored_components': metadata['components']
                }
                
                # Calculate retention status
                retention_until = datetime.fromisoformat(metadata['retention_until'])
                days_remaining = (retention_until - datetime.now()).days
                report['session_info']['days_until_deletion'] = max(0, days_remaining)
            else:
                report['session_info'] = {'status': 'not_found_or_expired'}
        
        else:
            # System-wide report
            try:
                session_files = [f for f in os.listdir('session_data') if f.endswith('_metadata.json')]
                total_sessions = len(session_files)
                
                active_sessions = 0
                expired_sessions = 0
                
                for metadata_file in session_files:
                    session_id = metadata_file.replace('_metadata.json', '')
                    if self._is_session_valid(session_id):
                        active_sessions += 1
                    else:
                        expired_sessions += 1
                
                report['system_info'] = {
                    'total_sessions': total_sessions,
                    'active_sessions': active_sessions,
                    'expired_sessions': expired_sessions,
                    'cleanup_needed': expired_sessions > 0
                }
                
            except Exception as e:
                report['system_info'] = {'error': f"Could not generate system report: {e}"}
        
        return report
    
    def export_user_data(self, session_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        
        export_result = {
            'export_timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'data': {},
            'metadata': {}
        }
        
        try:
            if self._is_session_valid(session_id):
                # Get session data
                session_data = self.retrieve_session_data(session_id)
                export_result['data'] = session_data
                
                # Get metadata
                metadata = self._load_session_metadata(session_id)
                export_result['metadata'] = metadata
                
                self.audit_logger.info(f"User data exported: {session_id}")
            else:
                export_result['error'] = 'Session not found or expired'
                
        except Exception as e:
            export_result['error'] = str(e)
            self.audit_logger.error(f"Data export failed: {session_id}, error={e}")
        
        return export_result
    
    def update_privacy_settings(self, session_id: str, new_privacy_level: str) -> Dict[str, Any]:
        """Update privacy settings for existing session"""
        
        update_result = {
            'session_id': session_id,
            'update_timestamp': datetime.now().isoformat(),
            'old_privacy_level': None,
            'new_privacy_level': new_privacy_level
        }
        
        try:
            if self._is_session_valid(session_id):
                metadata = self._load_session_metadata(session_id)
                update_result['old_privacy_level'] = metadata['privacy_level']
                
                # Update metadata
                metadata['privacy_level'] = new_privacy_level
                metadata['last_updated'] = update_result['update_timestamp']
                
                # Update retention period if needed
                new_settings = self.privacy_levels.get(new_privacy_level, self.privacy_levels['standard'])
                new_retention = datetime.now() + timedelta(days=new_settings['retention_days'])
                metadata['retention_until'] = new_retention.isoformat()
                
                # Save updated metadata
                self._save_session_metadata(session_id, metadata)
                
                self.audit_logger.info(
                    f"Privacy settings updated: {session_id}, "
                    f"{update_result['old_privacy_level']} -> {new_privacy_level}"
                )
            else:
                update_result['error'] = 'Session not found or expired'
                
        except Exception as e:
            update_result['error'] = str(e)
            self.audit_logger.error(f"Privacy settings update failed: {session_id}, error={e}")
        
        return update_result


# Example usage and testing
if __name__ == "__main__":
    # Test the privacy manager
    pm = PrivacyManager()
    
    print("Privacy Manager Test")
    print("=" * 50)
    
    # Generate session ID
    session_id = pm.generate_session_id()
    print(f"Generated Session ID: {session_id}")
    
    # Test data anonymization
    test_text = "Hi, my name is John Doe and my email is john.doe@email.com. My phone is 555-123-4567."
    anonymized = pm.anonymize_text(test_text, session_id)
    print(f"\nOriginal: {test_text}")
    print(f"Anonymized: {anonymized['anonymized_text']}")
    print(f"PII Found: {anonymized['pii_types_found']}")
    
    # Test data encryption
    test_data = {"message": "This is sensitive data", "emotion": "sad", "confidence": 0.8}
    encrypted = pm.encrypt_data(test_data, 'analysis')
    print(f"\nEncrypted data hash: {encrypted['data_hash']}")
    
    # Test decryption
    decrypted = pm.decrypt_data(encrypted)
    print(f"Decrypted: {decrypted}")
    
    # Test session storage
    session_data = {
        'text': test_text,
        'audio': {'emotion': 'neutral', 'confidence': 0.6},
        'video': {'emotion': 'sad', 'confidence': 0.7}
    }
    
    storage_result = pm.store_session_data(session_id, session_data, 'standard')
    print(f"\nStorage result: {storage_result['stored_components']}")
    
    # Test data retrieval
    retrieved = pm.retrieve_session_data(session_id, 'text')
    print(f"Retrieved text length: {len(retrieved.get('text', ''))}")
    
    # Test privacy report
    report = pm.get_privacy_report(session_id)
    print(f"\nPrivacy report: {report['session_info']['privacy_level']}")
    print(f"Days until deletion: {report['session_info']['days_until_deletion']}")
    
    # Test cleanup (won't delete recent session)
    cleanup_result = pm.cleanup_expired_sessions()
    print(f"\nCleanup: {len(cleanup_result['sessions_deleted'])} sessions deleted")