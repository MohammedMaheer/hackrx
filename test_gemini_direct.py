import os
import httpx
import asyncio
import json

async def test_gemini_api():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found in environment variables")
        return
    
    print(f"API Key: {GEMINI_API_KEY}")
    print(f"API URL: {GEMINI_API_URL}")
    
    # Simple test payload
    payload = {
        "contents": [{
            "parts": [{
                "text": "Say hello world"
            }]
        }]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                    print(f"Generated Content: {content}")
                else:
                    print("No candidates in response")
            else:
                print(f"Error: {response.status_code}")
                
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini_api())
