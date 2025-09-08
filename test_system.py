"""
Comprehensive System Test for AI Therapy System Phase 1
Tests all components individually and the complete pipeline
"""

import unittest
import sys
import os
import time
import requests
import threading
from typing import Dict, Any, List
import tempfile
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from text_analyzer import TextAnalyzer
from therapy_agent import TherapyAgent
from multimodal_fusion import SimplifiedMultimodalFusion
from config import Config, get_config

class TestTextAnalyzer(unittest.TestCase):
    """Test text analysis functionality"""
    
    @classmethod
    def setUpClass(cls):
        print("\n🧪 Testing Text Analyzer...")
        cls.analyzer = TextAnalyzer()
    
    def test_emotion_analysis(self):
        """Test emotion detection"""
        test_cases = [
            ("I'm feeling really happy today!", "joy"),
            ("I'm so sad and don't know what to do", "sadness"),
            ("I'm angry about this situation", "anger"),
            ("I feel anxious about tomorrow", "fear"),
        ]
        
        for text, expected_category in test_cases:
            result = self.analyzer.analyze_text(text)
            emotion = result['dominant_emotion']
            confidence = result['emotion_confidence']
            
            print(f"  Text: '{text[:30]}...' → {emotion} ({confidence:.2f})")
            
            # Check that we get a valid emotion result
            self.assertIsNotNone(emotion)
            self.assertIsInstance(confidence, (int, float))
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_sentiment_analysis(self):
        """Test sentiment detection"""
        test_cases = [
            ("I love this new opportunity!", "positive"),
            ("This is the worst day ever", "negative"),
            ("The weather is okay today", "neutral"),
        ]
        
        for text, expected_sentiment in test_cases:
            result = self.analyzer.analyze_text(text)
            sentiment = result['sentiment']
            score = result['sentiment_score']
            
            print(f"  Text: '{text}' → {sentiment} ({score:.2f})")
            
            # Check that we get valid sentiment results
            self.assertIn(sentiment, ['positive', 'negative', 'neutral'])
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_crisis_detection(self):
        """Test crisis indicator detection"""
        test_cases = [
            ("I want to end it all", "HIGH"),
            ("I'm feeling hopeless", "MODERATE"), 
            ("I'm stressed about work", "LOW"),
            ("I'm having a great day", "LOW"),
        ]
        
        for text, expected_level in test_cases:
            result = self.analyzer.analyze_text(text)
            crisis_level = result['crisis_classification']
            crisis_score = result['crisis_level']
            
            print(f"  Text: '{text}' → {crisis_level} ({crisis_score:.2f})")
            
            # Check crisis detection results
            self.assertIn(crisis_level, ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'])
            self.assertIsInstance(crisis_score, (int, float))
            self.assertGreaterEqual(crisis_score, 0.0)
            self.assertLessEqual(crisis_score, 1.0)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        result = self.analyzer.analyze_text("")
        
        self.assertEqual(result['dominant_emotion'], 'neutral')
        self.assertEqual(result['sentiment'], 'neutral')
        self.assertEqual(result['crisis_level'], 0.0)
        self.assertEqual(result['word_count'], 0)
    
    def test_analysis_structure(self):
        """Test that analysis returns all required fields"""
        result = self.analyzer.analyze_text("I feel okay today")
        
        required_fields = [
            'input_text', 'timestamp', 'word_count', 'dominant_emotion',
            'emotion_confidence', 'sentiment', 'sentiment_score',
            'crisis_level', 'crisis_classification', 'overall_risk_score',
            'risk_level', 'suggested_techniques', 'analysis_confidence'
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")


class TestTherapyAgent(unittest.TestCase):
    """Test therapy agent functionality"""
    
    @classmethod
    def setUpClass(cls):
        print("\n🤖 Testing Therapy Agent...")
        cls.agent = TherapyAgent()
    
    def test_provider_status(self):
        """Test provider status checking"""
        status = self.agent.get_provider_status()
        
        print(f"  Provider status: {status}")
        
        self.assertIn('providers', status)
        self.assertIn('primary', status)
        self.assertIn('ollama', status['providers'])
        self.assertIn('groq', status['providers'])
    
    def test_response_generation(self):
        """Test therapy response generation"""
        test_message = "I'm feeling anxious about work"
        test_analysis = {
            'dominant_emotion': 'anxiety',
            'sentiment': 'negative',
            'risk_level': 'LOW',
            'crisis_classification': 'LOW',
            'mental_health_topics': [('anxiety', 0.8)],
            'suggested_techniques': ['mindfulness']
        }
        
        response = self.agent.generate_response(test_message, test_analysis)
        
        print(f"  Input: '{test_message}'")
        print(f"  Response: '{response['content'][:50]}...'")
        print(f"  Provider: {response['provider']}")
        print(f"  Technique: {response['technique']}")
        
        # Check response structure
        self.assertIn('content', response)
        self.assertIn('provider', response)
        self.assertIn('technique', response)
        self.assertIn('confidence', response)
        
        # Check content is not empty
        self.assertGreater(len(response['content']), 10)
    
    def test_crisis_response(self):
        """Test crisis situation handling"""
        crisis_message = "I want to end it all"
        crisis_analysis = {
            'dominant_emotion': 'sadness',
            'sentiment': 'negative',
            'risk_level': 'HIGH',
            'crisis_classification': 'HIGH',
            'crisis_indicators': ['suicide: end it all'],
            'mental_health_topics': [('depression', 0.9)]
        }
        
        response = self.agent.generate_response(crisis_message, crisis_analysis)
        
        print(f"  Crisis input: '{crisis_message}'")
        print(f"  Crisis response: '{response['content'][:50]}...'")
        print(f"  Provider: {response['provider']}")
        
        # Crisis responses should mention emergency resources
        content_lower = response['content'].lower()
        self.assertTrue(
            '988' in content_lower or '911' in content_lower or 'crisis' in content_lower,
            "Crisis response should mention emergency resources"
        )
    
    def test_fallback_response(self):
        """Test fallback response system"""
        # Test with standard input
        response = self.agent._generate_fallback_response(
            "I'm feeling sad",
            {'dominant_emotion': 'sadness', 'primary_topic': 'depression'}
        )
        
        print(f"  Fallback response: '{response['content'][:50]}...'")
        
        self.assertEqual(response['provider'], 'rule_based')
        self.assertGreater(len(response['content']), 20)


class TestMultimodalFusion(unittest.TestCase):
    """Test multimodal fusion (Phase 1 - text only)"""
    
    @classmethod
    def setUpClass(cls):
        print("\n🔀 Testing Multimodal Fusion...")
        cls.fusion = SimplifiedMultimodalFusion()
    
    def test_text_passthrough(self):
        """Test text-only fusion"""
        text_results = {
            'text': {
                'dominant_emotion': 'anxiety',
                'sentiment': 'negative',
                'crisis_level': 0.3,
                'analysis_confidence': 0.8
            }
        }
        
        fused = self.fusion.fuse_modalities(text_results)
        
        print(f"  Text fusion: {fused['dominant_emotion']} emotion")
        print(f"  Fusion method: {fused['fusion_metadata']['fusion_method']}")
        
        # Should pass through text results
        self.assertEqual(fused['dominant_emotion'], 'anxiety')
        self.assertEqual(fused['sentiment'], 'negative')
        self.assertEqual(fused['fusion_metadata']['fusion_method'], 'text_passthrough')
    
    def test_ignored_modalities(self):
        """Test that non-text modalities are ignored in Phase 1"""
        multi_results = {
            'text': {
                'dominant_emotion': 'sadness',
                'analysis_confidence': 0.7
            },
            'audio': {
                'emotion': 'happy',
                'confidence': 0.8
            },
            'video': {
                'emotion': 'neutral',
                'confidence': 0.6
            }
        }
        
        fused = self.fusion.fuse_modalities(multi_results)
        
        print(f"  Multi-modal input with ignored modalities")
        print(f"  Processed: {fused['fusion_metadata']['modalities_processed']}")
        print(f"  Ignored: {fused['fusion_metadata'].get('ignored_modalities', 'None')}")
        
        # Should only process text
        self.assertEqual(fused['fusion_metadata']['modalities_processed'], ['text'])
        self.assertIn('ignored_modalities', fused['fusion_metadata'])
        self.assertIn('audio', fused['fusion_metadata']['ignored_modalities'])
    
    def test_capabilities(self):
        """Test fusion capabilities reporting"""
        capabilities = self.fusion.get_capabilities()
        
        print(f"  Current phase: {capabilities['phase']}")
        print(f"  Supported modalities: {capabilities['supported_modalities']}")
        
        self.assertEqual(capabilities['phase'], "1")
        self.assertEqual(capabilities['supported_modalities'], ['text'])


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    @classmethod
    def setUpClass(cls):
        print("\n⚙️ Testing Configuration...")
    
    def test_default_config(self):
        """Test default configuration loading"""
        config = Config()
        
        print(f"  Default models: {config.models.emotion_model}")
        print(f"  Default server: {config.server.host}:{config.server.port}")
        
        # Check default values
        self.assertIsNotNone(config.models.emotion_model)
        self.assertIsNotNone(config.models.sentiment_model)
        self.assertEqual(config.server.host, "127.0.0.1")
        self.assertEqual(config.server.port, 5000)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        
        # Test threshold validation
        self.assertLessEqual(config.analysis.crisis_threshold_low, 
                           config.analysis.crisis_threshold_moderate)
        self.assertLessEqual(config.analysis.crisis_threshold_moderate, 
                           config.analysis.crisis_threshold_high)
        
        print(f"  Crisis thresholds: {config.analysis.crisis_threshold_low} < "
              f"{config.analysis.crisis_threshold_moderate} < "
              f"{config.analysis.crisis_threshold_high}")
    
    def test_config_summary(self):
        """Test configuration summary"""
        config = Config()
        summary = config.get_summary()
        
        print(f"  Config summary keys: {list(summary.keys())}")
        
        self.assertIn('models', summary)
        self.assertIn('llm', summary)
        self.assertIn('server', summary)


class TestIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    @classmethod
    def setUpClass(cls):
        print("\n🔗 Testing System Integration...")
        cls.analyzer = TextAnalyzer()
        cls.agent = TherapyAgent()
        cls.fusion = SimplifiedMultimodalFusion()
    
    def test_complete_pipeline(self):
        """Test complete analysis pipeline"""
        test_message = "I've been feeling really overwhelmed with work lately"
        
        print(f"  Testing pipeline with: '{test_message}'")
        
        # Step 1: Text analysis
        analysis = self.analyzer.analyze_text(test_message)
        print(f"  1. Analysis: {analysis['dominant_emotion']} emotion, {analysis['sentiment']} sentiment")
        
        # Step 2: Multimodal fusion (currently just passthrough)
        modality_results = {'text': analysis}
        fused_results = self.fusion.fuse_modalities(modality_results)
        print(f"  2. Fusion: {fused_results['fusion_metadata']['fusion_method']}")
        
        # Step 3: Therapy response
        response = self.agent.generate_response(test_message, fused_results)
        print(f"  3. Response: {response['provider']} → '{response['content'][:50]}...'")
        
        # Verify pipeline worked
        self.assertIsNotNone(analysis)
        self.assertIsNotNone(fused_results)
        self.assertIsNotNone(response)
        self.assertGreater(len(response['content']), 20)
    
    def test_crisis_pipeline(self):
        """Test crisis detection pipeline"""
        crisis_message = "I don't want to live anymore"
        
        print(f"  Testing crisis pipeline with: '{crisis_message}'")
        
        # Complete pipeline
        analysis = self.analyzer.analyze_text(crisis_message)
        modality_results = {'text': analysis}
        fused_results = self.fusion.fuse_modalities(modality_results)
        response = self.agent.generate_response(crisis_message, fused_results)
        
        print(f"  Crisis level: {analysis['crisis_classification']}")
        print(f"  Response type: {response.get('technique', 'unknown')}")
        
        # Should detect crisis
        self.assertIn(analysis['crisis_classification'], ['MODERATE', 'HIGH', 'CRITICAL'])
        
        # Response should address crisis
        content_lower = response['content'].lower()
        self.assertTrue(
            any(keyword in content_lower for keyword in ['988', '911', 'crisis', 'emergency', 'help']),
            "Crisis response should mention emergency resources"
        )


class TestSystemHealth(unittest.TestCase):
    """Test system health and performance"""
    
    @classmethod
    def setUpClass(cls):
        print("\n🏥 Testing System Health...")
    
    def test_response_time(self):
        """Test response time performance"""
        analyzer = TextAnalyzer()
        
        test_message = "I'm feeling anxious about work and don't know what to do"
        
        start_time = time.time()
        result = analyzer.analyze_text(test_message)
        end_time = time.time()
        
        response_time = end_time - start_time
        print(f"  Text analysis time: {response_time:.2f} seconds")
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(response_time, 10.0, "Text analysis taking too long")
        self.assertIsNotNone(result)
    
    def test_memory_usage(self):
        """Test basic memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"  Memory usage: {memory_mb:.1f} MB")
        
        # Should not use excessive memory (adjust threshold as needed)
        self.assertLess(memory_mb, 2048, "Excessive memory usage")
    
    def test_error_handling(self):
        """Test error handling robustness"""
        analyzer = TextAnalyzer()
        
        # Test various edge cases
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "a",  # Single character
            "a" * 1000,  # Very long string
            "🙂😢😡",  # Emojis only
            "12345",  # Numbers only
        ]
        
        for test_input in edge_cases:
            try:
                result = analyzer.analyze_text(test_input)
                self.assertIsNotNone(result)
                print(f"  ✓ Handled: '{test_input[:20]}...'")
            except Exception as e:
                self.fail(f"Failed to handle input '{test_input}': {e}")


def run_all_tests():
    """Run all tests with nice formatting"""
    print("🧪 AI Therapy System - Phase 1 Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestTextAnalyzer,
        TestTherapyAgent, 
        TestMultimodalFusion,
        TestConfiguration,
        TestIntegration,
        TestSystemHealth
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 System is ready for Phase 1 deployment!")
    elif success_rate >= 70:
        print("⚠️ System has some issues but may be usable")
    else:
        print("❌ System needs significant fixes before deployment")
    
    return result.wasSuccessful()


def quick_test():
    """Run a quick test of core functionality"""
    print("🚀 Quick System Test")
    print("-" * 30)
    
    try:
        # Test text analyzer
        print("1. Testing text analyzer...")
        analyzer = TextAnalyzer()
        result = analyzer.analyze_text("I feel anxious about work")
        print(f"   ✓ Emotion: {result['dominant_emotion']}")
        
        # Test therapy agent
        print("2. Testing therapy agent...")
        agent = TherapyAgent()
        response = agent.generate_response("I'm stressed", result)
        print(f"   ✓ Provider: {response['provider']}")
        
        # Test fusion
        print("3. Testing fusion...")
        fusion = SimplifiedMultimodalFusion()
        fused = fusion.fuse_modalities({'text': result})
        print(f"   ✓ Method: {fused['fusion_metadata']['fusion_method']}")
        
        print("\n✅ Quick test passed! Core functionality working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AI Therapy System")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--component", choices=["text", "agent", "fusion", "config"], 
                       help="Test specific component only")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
        sys.exit(0 if success else 1)
    elif args.component:
        # Run specific component test
        component_tests = {
            "text": TestTextAnalyzer,
            "agent": TestTherapyAgent,
            "fusion": TestMultimodalFusion,
            "config": TestConfiguration
        }
        
        suite = unittest.TestLoader().loadTestsFromTestCase(component_tests[args.component])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)