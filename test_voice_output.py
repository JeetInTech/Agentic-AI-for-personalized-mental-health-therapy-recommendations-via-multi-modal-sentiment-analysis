#!/usr/bin/env python3
"""
Simple test script for voice output
"""

import time
from voice_agent import VoiceAgent

print("Testing Voice Output...")
print("=" * 30)

# Create voice agent
voice_agent = VoiceAgent()

# Test multiple TTS calls to ensure no run loop conflicts
test_messages = [
    "Hello, this is test message one.",
    "This is test message two.",
    "Final test message three."
]

for i, message in enumerate(test_messages, 1):
    print(f"\n{i}. Testing: '{message}'")
    result = voice_agent.speak_text(message)

    if result['success']:
        print(f"   [OK] Queued successfully (Queue size: {result['queue_size']})")
    else:
        print(f"   [ERROR] Failed: {result['error']}")

    # Short pause between messages
    time.sleep(2)

print("\nWaiting for speech to complete...")
time.sleep(5)

# Test status
status = voice_agent.get_voice_status()
print(f"\nVoice Status:")
print(f"  - TTS Available: {status['text_to_speech_available']}")
print(f"  - Is Speaking: {status['is_speaking']}")

print("\n[SUCCESS] Voice output test completed!")