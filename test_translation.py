#!/usr/bin/env python3
"""
Test script for the translation backend
Run this to verify the server is working correctly
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is running!")
            print(f"  - CUDA available: {data.get('cuda_available')}")
            print(f"  - Device: {data.get('device')}")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is it running?")
        print("  Run: cd backend && ./start.sh")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_translation(text, src, tgt, expected_contains=None):
    """Test a translation request"""
    print(f"\n📝 Testing translation: {src} → {tgt}")
    print(f"   Input: {text}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/translation/v2",
            json={"text": text, "src": src, "tgt": tgt},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()["result"]
            print(f"   Output: {result}")
            
            if expected_contains and expected_contains.lower() not in result.lower():
                print(f"   ⚠️  Expected to contain '{expected_contains}' but didn't")
            else:
                print(f"   ✓ Translation successful!")
            return True
        else:
            print(f"   ✗ Translation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"   ✗ Translation timed out (>30s)")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Sami Translation Backend - Test Suite")
    print("=" * 60)
    
    # Test health check
    if not test_health_check():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Running translation tests...")
    print("=" * 60)
    
    # Test Sami → Norwegian
    test_translation(
        "Bures!",
        "sme",
        "nor",
        expected_contains="hei"
    )
    
    test_translation(
        "Mun lean duohta.",
        "sme",
        "nor",
        expected_contains="jeg"
    )
    
    # Test Norwegian → Sami
    test_translation(
        "Hei! Hvordan har du det?",
        "nor",
        "sme",
        expected_contains="Bures"
    )
    
    test_translation(
        "Takk for hjelpen!",
        "nor",
        "sme",
        expected_contains="giitu"
    )
    
    # Test longer text
    test_translation(
        "Davvisámegiella lea sámegiela stuorámus ja eanet go 20000 olmmái eatnigiella. Sámegiela guovdu gullá Guovdageainnu, Kárášjohka ja Roavvenjárgga suohkanat, muhto davvisápmelaččat orrot maiddái eará Norgga ja Suoma guovlluin.",
        "sme",
        "nor"
    )
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
