#!/usr/bin/env python3
"""
Test script for LLM APIs (Perplexity and Gemini fallback)
"""

import asyncio
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_explainer import generate_explainable_answer

async def test_answer_generation():
    """Test answer generation with both Perplexity and Gemini fallback"""
    
    # Test with sample data
    question = "What is the main purpose of this document?"
    
    # Sample document chunks (dummy data for testing)
    sample_chunks = [
        "This agreement outlines the terms and conditions for insurance coverage. The policy holder agrees to pay premiums on a monthly basis.",
        "The insurance company will provide coverage for damages up to the policy limit. Claims must be filed within 30 days of incident occurrence.",
        "Exclusions to this policy include acts of war, nuclear events, and intentional self-harm. Coverage does not extend to these circumstances.",
        "Premium payments are due on the first day of each month. Late payments may result in policy cancellation after 30 days.",
        "The policy term is one year from the effective date. Renewal notices will be sent 30 days prior to expiration."
    ]
    
    print("=" * 60)
    print("Testing Answer Generation Pipeline")
    print("=" * 60)
    
    # Test answer generation
    result = await generate_explainable_answer(question, sample_chunks)
    
    print(f"Question: {question}")
    print("\nGenerated Answer:")
    print(json.dumps(result, indent=2))
    
    # Check if we got a valid answer
    if "Unable to generate answer" in result.get("answer", ""):
        print("\n❌ Answer generation failed")
        if "Perplexity" in result.get("answer", ""):
            print("   - Perplexity API error detected")
        if "Gemini" in result.get("answer", ""):
            print("   - Gemini API error detected")
    else:
        print("\n✅ Answer generation successful")
        print(f"   - Answer: {result.get('answer', '')[:100]}...")
        print(f"   - Confidence: {result.get('confidence', 0)}")

async def main():
    """Main test function"""
    await test_answer_generation()

if __name__ == "__main__":
    asyncio.run(main())
