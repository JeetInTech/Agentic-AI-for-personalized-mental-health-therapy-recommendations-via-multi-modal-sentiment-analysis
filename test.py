#!/usr/bin/env python3
"""
Ollama API Testing Script
Run this to debug the exact issue with Ollama integration
"""

import requests
import json
import time

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("=" * 50)
    print("STEP 1: Testing Ollama Connection")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print("Available models:")
            for model in data.get('models', []):
                print(f"  - {model.get('name', 'Unknown')}")
            return True
        else:
            print("ERROR: Ollama not responding correctly")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_model_generation():
    """Test model generation with different formats"""
    print("\n" + "=" * 50)
    print("STEP 2: Testing Model Generation")
    print("=" * 50)
    
    # Test cases with different model names and formats
    test_cases = [
        {
            "name": "Test 1 - llama3.1:8b",
            "model": "llama3.1:8b",
            "endpoint": "/api/generate"
        },
        {
            "name": "Test 2 - llama3.1",
            "model": "llama3.1",
            "endpoint": "/api/generate"
        },
        {
            "name": "Test 3 - Chat endpoint",
            "model": "llama3.1:8b",
            "endpoint": "/v1/chat/completions"
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print("-" * 30)
        
        if test['endpoint'] == '/api/generate':
            test_generate_endpoint(test['model'])
        else:
            test_chat_endpoint(test['model'])

def test_generate_endpoint(model_name):
    """Test /api/generate endpoint"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": "Hello, how are you?",
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Response: {data.get('response', 'No response field')[:100]}...")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_chat_endpoint(model_name):
    """Test /v1/chat/completions endpoint"""
    url = "http://localhost:11434/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"Success! Response: {content[:100]}...")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_exact_therapy_request():
    """Test the exact request format from therapy_agent.py"""
    print("\n" + "=" * 50)
    print("STEP 3: Testing Exact Therapy Agent Format")
    print("=" * 50)
    
    url = "http://localhost:11434/api/generate"
    
    # Simulate the exact context from therapy_agent
    context = """You are a compassionate, professional mental health therapist. Respond to the user with empathy and therapeutic techniques.

Analysis of user's message:
- Emotion: sadness
- Sentiment: negative
- Risk level: LOW
- Topics: relationship
- Suggested techniques: supportive_counseling

User's message: "I broke up recently"

Provide a therapeutic response that:
1. Acknowledges their feelings with empathy
2. Uses appropriate therapeutic techniques
3. Is supportive but not giving medical advice
4. Encourages professional help if needed
5. Keeps the response concise (2-3 sentences)

Therapeutic response:"""
    
    payload = {
        "model": "llama3.1:8b",
        "prompt": context,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 300
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"URL: {url}")
    print(f"Model: {payload['model']}")
    print(f"Options: {payload['options']}")
    print("Testing...")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            therapeutic_response = data.get('response', '').strip()
            print(f"Success!")
            print(f"Therapeutic Response: {therapeutic_response}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_request_variations():
    """Test variations to find what works"""
    print("\n" + "=" * 50)
    print("STEP 4: Testing Request Variations")
    print("=" * 50)
    
    base_url = "http://localhost:11434/api/generate"
    
    variations = [
        {
            "name": "Minimal request",
            "payload": {
                "model": "llama3.1:8b",
                "prompt": "Hello"
            }
        },
        {
            "name": "With stream=false",
            "payload": {
                "model": "llama3.1:8b", 
                "prompt": "Hello",
                "stream": False
            }
        },
        {
            "name": "With options",
            "payload": {
                "model": "llama3.1:8b",
                "prompt": "Hello", 
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            }
        },
        {
            "name": "Different model format",
            "payload": {
                "model": "llama3.1",
                "prompt": "Hello",
                "stream": False
            }
        }
    ]
    
    for variation in variations:
        print(f"\n{variation['name']}:")
        print("-" * 20)
        
        try:
            response = requests.post(
                base_url, 
                json=variation['payload'], 
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data.get('response', 'No response')[:50]}...")
            else:
                print(f"Error: {response.text[:100]}...")
                
        except Exception as e:
            print(f"Exception: {e}")

def main():
    """Run all tests"""
    print("OLLAMA API DEBUGGING SCRIPT")
    print("Testing step by step to find the issue...")
    
    # Step 1: Test connection
    if not test_ollama_connection():
        print("\nFAILED: Cannot connect to Ollama")
        return
    
    # Step 2: Test different formats
    test_model_generation()
    
    # Step 3: Test exact therapy format
    test_exact_therapy_request()
    
    # Step 4: Test variations
    test_request_variations()
    
    print("\n" + "=" * 50)
    print("DEBUGGING COMPLETE")
    print("=" * 50)
    print("Check the results above to see which format works.")
    print("Then update your therapy_agent.py accordingly.")

if __name__ == "__main__":
    main()