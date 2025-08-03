#!/usr/bin/env python3
"""
Test script for the Railway deployment
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "https://hackrx-production-f2ba.up.railway.app"
API_KEY = os.getenv("HACKRX_API_KEY", "bfb8fabaf1ce137c1402366fb3d5a052836234c1ff376c326842f52e3164cc33")

# Test data
TEST_DOCUMENT_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
TEST_QUESTIONS = ["test"]

async def test_main_endpoint():
    """Test the main /hackrx/run endpoint"""
    print("Testing main endpoint...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            print(f"Sending request to {BASE_URL}/hackrx/run")
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=payload
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success! Got {len(data.get('answers', []))} answers")
                return True
            else:
                print(f"Error response: {response.text}")
                return False
                
        except httpx.TimeoutException:
            print("Request timeout")
            return False
        except Exception as e:
            print(f"Request failed: {e}")
            return False

async def main():
    """Run the test"""
    print("=" * 50)
    print("Railway Deployment Test")
    print("=" * 50)
    
    main_ok = await test_main_endpoint()
    
    if main_ok:
        print("Main endpoint test passed!")
    else:
        print("Main endpoint test failed!")

if __name__ == "__main__":
    asyncio.run(main())
