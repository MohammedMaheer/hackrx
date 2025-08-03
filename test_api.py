#!/usr/bin/env python3
"""
Test script for the HackRx LLM Query-Retrieval System
"""

import asyncio
import pytest
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "https://hackrx-production-f2ba.up.railway.app"  # Change this to your Railway URL when deployed
API_KEY = os.getenv("HACKRX_API_KEY", "bfb8fabaf1ce137c1402366fb3d5a052836234c1ff376c326842f52e3164cc33")

# Test data
TEST_DOCUMENT_URL = "https://irdai.gov.in/documents/37343/993134/75.All+Risk+-+Policy+Wording_GEN709.pdf/b27ce589-4588-1ea9-34d2-4da9c18fc5f0?version=1.1&t=1668340245453&download=true"  # Real IRDAI insurance policy PDF
TEST_QUESTIONS = [
    "What is the coverage amount for medical expenses?",
    "What are the exclusions mentioned in this policy?",
    "What is the claim procedure?",
    "What is the premium payment schedule?"
]

@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            print(f"Health check status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            import traceback
            print("Health check failed:")
            traceback.print_exc()
            return False

@pytest.mark.asyncio
async def test_main_endpoint():
    """Test the main /hackrx/run endpoint"""
    print("\nTesting main endpoint...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            print(f"Sending request to {BASE_URL}/hackrx/run")
            print(f"Document: {TEST_DOCUMENT_URL}")
            print(f"Questions: {len(TEST_QUESTIONS)}")
            
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=payload
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success! Got {len(data.get('answers', []))} answers")
                
                # Print first answer as example
                if data.get("answers"):
                    first_answer = data["answers"][0]
                    print(f"\nFirst Answer:")
                    print(f"Question: {TEST_QUESTIONS[0]}")
                    print(f"Answer: {first_answer.get('answer', '')[:200]}...")
                    print(f"Confidence: {first_answer.get('confidence_score', 0)}")
                    print(f"Matched clauses: {len(first_answer.get('matched_clauses', []))}")
                
                return True
            else:
                print(f"Error response: {response.text}")
                return False
                
        except httpx.TimeoutException:
            print("Request timeout - the server might be processing a large document")
            return False
        except Exception as e:
            print(f"Request failed: {e}")
            return False

@pytest.mark.asyncio
async def test_authentication():
    """Test authentication with invalid token"""
    print("\nTesting authentication...")
    
    headers = {
        "Authorization": "Bearer invalid-token",
        "Content-Type": "application/json"
    }
    
    data = {
        "documents": TEST_DOCUMENT_URL,
        "questions": ["Test question"]
    }
    
    # Set a timeout of 5 minutes (300 seconds) for the entire request
    timeout = httpx.Timeout(300.0, connect=60.0)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            print(f"Sending request to {BASE_URL}/hackrx/run")
            print(f"Document: {TEST_DOCUMENT_URL}")
            print(f"Questions: {len(['Test question'])}")
            
            # Log the request
            print("\nRequest Headers:")
            for key, value in headers.items():
                if key.lower() == 'authorization':
                    print(f"{key}: Bearer {'*' * 10}")
                else:
                    print(f"{key}: {value}")
            
            print("\nRequest Body:")
            print(json.dumps(data, indent=2))
            
            # Make the request
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=data
            )
            
            print(f"\nResponse Status: {response.status_code}")
            print("Response Headers:")
            for key, value in response.headers.items():
                print(f"{key}: {value}")
                
            print("\nResponse Body:")
            try:
                response_data = response.json()
                print(json.dumps(response_data, indent=2))
                
                # Check response status code
                if response.status_code == 401:
                    print("Authentication test passed - invalid token rejected")
                    return True
                print(f"Unexpected status code: {response.status_code}")
                return False
                    
            except json.JSONDecodeError as e:
                print(f"Could not parse JSON response: {e}")
                print(f"Raw response: {response.text[:1000]}...")
                return False
            except Exception as e:
                print(f"Error processing response: {e}")
                return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 50)
    print("HackRx LLM Query-Retrieval System Test")
    print("=" * 50)
    
    print(f"Testing against: {BASE_URL}")
    print(f"Using API key: {API_KEY[:20]}...")
    
    # Run tests
    health_ok = await test_health_check()
    auth_ok = await test_authentication()
    
    if health_ok:
        main_ok = await test_main_endpoint()
    else:
        print("Skipping main endpoint test due to health check failure")
        main_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Health Check: {'✓' if health_ok else '✗'}")
    print(f"Authentication: {'✓' if auth_ok else '✗'}")
    print(f"Main Endpoint: {'✓' if main_ok else '✗'}")
    
    if all([health_ok, auth_ok, main_ok]):
        print("\n🎉 All tests passed! Your API is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
    
    return all([health_ok, auth_ok, main_ok])

if __name__ == "__main__":
    # Allow overriding base URL via command line
    import sys
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
        print(f"Using custom base URL: {BASE_URL}")
    
    success = asyncio.run(main())
    exit(0 if success else 1)