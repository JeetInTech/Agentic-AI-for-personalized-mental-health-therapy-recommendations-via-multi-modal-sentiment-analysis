"""
Simplified Multimodal Fusion for Phase 1
Pass-through system that only handles text analysis
Audio/Video fusion will be added in Phase 2 and 3
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SimplifiedMultimodalFusion:
    """
    Phase 1: Simplified fusion that only passes through text analysis
    Phase 2+: Will add audio and video fusion capabilities
    """
    
    def __init__(self):
        self.phase = "1"
        self.supported_modalities = ["text"]
        logger.info("Initialized simplified multimodal fusion (Phase 1 - text only)")
    
    def fuse_modalities(self, modality_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 1: Pass through text analysis results only
        Phase 2+: Will implement full multimodal fusion
        
        Args:
            modality_results: Dictionary containing results from different modalities
            
        Returns:
            Fused analysis results (currently just text results)
        """
        
        # Phase 1: Only handle text modality
        if 'text' in modality_results:
            text_results = modality_results['text']
            
            # Add fusion metadata
            fused_results = text_results.copy()
            fused_results.update({
                'fusion_metadata': {
                    'phase': self.phase,
                    'modalities_processed': ['text'],
                    'modalities_available': list(modality_results.keys()),
                    'fusion_method': 'text_passthrough',
                    'fusion_confidence': text_results.get('analysis_confidence', 0.5),
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            # Log if other modalities were provided but not processed
            other_modalities = [k for k in modality_results.keys() if k != 'text']
            if other_modalities:
                logger.info(f"Phase 1: Ignoring {other_modalities} - text only processing")
                fused_results['fusion_metadata']['ignored_modalities'] = other_modalities
            
            return fused_results
        
        else:
            # No text analysis available - return empty result
            logger.warning("No text analysis available for fusion")
            return self._empty_fusion_result(modality_results)
    
    def _empty_fusion_result(self, modality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Return empty fusion result when no text analysis is available"""
        return {
            'input_text': '',
            'dominant_emotion': 'neutral',
            'emotion_confidence': 0.0,
            'sentiment': 'neutral',
            'sentiment_score': 0.5,
            'crisis_level': 0.0,
            'crisis_classification': 'LOW',
            'overall_risk_score': 0.0,
            'risk_level': 'LOW',
            'analysis_confidence': 0.0,
            'fusion_metadata': {
                'phase': self.phase,
                'modalities_processed': [],
                'modalities_available': list(modality_results.keys()),
                'fusion_method': 'empty_fallback',
                'fusion_confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': 'no_text_analysis_available'
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return current fusion capabilities"""
        return {
            'phase': self.phase,
            'supported_modalities': self.supported_modalities,
            'description': 'Phase 1: Text-only passthrough fusion',
            'future_capabilities': [
                'Phase 2: Audio fusion',
                'Phase 3: Video fusion', 
                'Phase 3: Full multimodal fusion with temporal analysis'
            ]
        }
    
    def is_ready_for_phase_2(self) -> bool:
        """Check if system is ready to upgrade to Phase 2 (audio fusion)"""
        # This would check if audio analyzer is available and working
        return False
    
    def is_ready_for_phase_3(self) -> bool:
        """Check if system is ready to upgrade to Phase 3 (video fusion)"""
        # This would check if video analyzer is available and working
        return False


# Backward compatibility - maintain original class name
class MultimodalFusion(SimplifiedMultimodalFusion):
    """Backward compatibility alias"""
    pass


# Test function
def test_simplified_fusion():
    """Test the simplified fusion system"""
    fusion = SimplifiedMultimodalFusion()
    
    # Test case 1: Text analysis only
    test_text_results = {
        'text': {
            'input_text': 'I feel anxious about work',
            'dominant_emotion': 'anxiety',
            'emotion_confidence': 0.8,
            'sentiment': 'negative',
            'sentiment_score': 0.7,
            'crisis_level': 0.2,
            'crisis_classification': 'LOW',
            'overall_risk_score': 0.3,
            'risk_level': 'LOW',
            'analysis_confidence': 0.75
        }
    }
    
    print("Testing Simplified Multimodal Fusion")
    print("=" * 50)
    
    # Test text-only fusion
    print("Test 1: Text-only analysis")
    result = fusion.fuse_modalities(test_text_results)
    print(f"Emotion: {result['dominant_emotion']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Fusion method: {result['fusion_metadata']['fusion_method']}")
    print(f"Modalities processed: {result['fusion_metadata']['modalities_processed']}")
    print()
    
    # Test case 2: Multiple modalities (should ignore non-text)
    test_multi_results = {
        'text': {
            'dominant_emotion': 'sadness',
            'sentiment': 'negative',
            'analysis_confidence': 0.8
        },
        'audio': {
            'dominant_emotion': 'sad',
            'confidence': 0.7
        },
        'video': {
            'dominant_emotion': 'neutral',
            'confidence': 0.6
        }
    }
    
    print("Test 2: Multiple modalities (should ignore audio/video)")
    result = fusion.fuse_modalities(test_multi_results)
    print(f"Emotion: {result['dominant_emotion']}")
    print(f"Modalities available: {result['fusion_metadata']['modalities_available']}")
    print(f"Modalities processed: {result['fusion_metadata']['modalities_processed']}")
    print(f"Ignored modalities: {result['fusion_metadata'].get('ignored_modalities', 'None')}")
    print()
    
    # Test case 3: No text analysis
    test_no_text = {
        'audio': {'emotion': 'happy'},
        'video': {'emotion': 'neutral'}
    }
    
    print("Test 3: No text analysis available")
    result = fusion.fuse_modalities(test_no_text)
    print(f"Emotion: {result['dominant_emotion']}")
    print(f"Fusion method: {result['fusion_metadata']['fusion_method']}")
    print(f"Error: {result['fusion_metadata'].get('error', 'None')}")
    print()
    
    # Show capabilities
    print("Current capabilities:")
    capabilities = fusion.get_capabilities()
    print(f"Phase: {capabilities['phase']}")
    print(f"Supported: {capabilities['supported_modalities']}")
    print(f"Description: {capabilities['description']}")
    print("Future capabilities:")
    for capability in capabilities['future_capabilities']:
        print(f"  - {capability}")


if __name__ == "__main__":
    test_simplified_fusion()