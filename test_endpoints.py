#!/usr/bin/env python3
"""
Quick verification script to test that all documented endpoints exist and respond.
Run this after starting the server with: TEST_MODE=1 python main.py
"""
import requests
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, description):
    """Test an endpoint and report results"""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json={"text": "test", "src": "sme", "tgt": "nor"}, timeout=5)
        
        status = "✓" if response.status_code < 400 else "✗"
        print(f"{status} {method} {path} - {description}")
        print(f"  Status: {response.status_code}")
        if response.status_code >= 400:
            print(f"  Error: {response.text[:200]}")
        return response.status_code < 400
    except requests.exceptions.ConnectionError:
        print(f"✗ {method} {path} - {description}")
        print(f"  Error: Connection refused (server not running?)")
        return False
    except Exception as e:
        print(f"✗ {method} {path} - {description}")
        print(f"  Error: {e}")
        return False

def main():
    print("Testing Sami Translation Backend Endpoints")
    print("=" * 60)
    
    tests = [
        ("GET", "/", "Health check"),
        ("GET", "/translation/openapi.json", "OpenAPI specification"),
        ("GET", "/translation/docs", "Swagger UI"),
        ("GET", "/translation/redoc", "ReDoc UI"),
        ("GET", "/translation/v2", "Get capabilities (language pairs)"),
        ("POST", "/translation/v2", "Translate text"),
    ]
    
    results = []
    for method, path, description in tests:
        results.append(test_endpoint(method, path, description))
        print()
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} endpoints working")
    
    if passed == total:
        print("\n✓ All endpoints are correctly implemented!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} endpoint(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
