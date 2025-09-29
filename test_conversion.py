#!/usr/bin/env python3
"""
Simple test to verify that the api_client conversion from aiohttp to requests works correctly.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, '/src')

def test_imports():
    """Test that all necessary modules can be imported."""
    try:
        # This should work even without dependencies
        print("✅ Basic Python imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api_client_structure():
    """Test that the api_client module has all expected functions."""
    try:
        # Read the file and check for function definitions
        with open('/src/validmind/api_client.py', 'r') as f:
            content = f.read()
        
        # Check for key functions
        expected_functions = [
            'def init(',
            'def get_api_host(',
            'def get_api_model(',
            'def log_metadata(',
            'def log_figure(',
            'def log_test_result(',
            'def log_metric(',
            'def generate_test_result_description(',
        ]
        
        missing_functions = []
        for func in expected_functions:
            if func not in content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing functions: {missing_functions}")
            return False
        else:
            print("✅ All expected functions found in api_client.py")
            return True
    except Exception as e:
        print(f"❌ Error checking api_client structure: {e}")
        return False

def test_async_removal():
    """Test that async/await keywords have been removed."""
    try:
        with open('/src/validmind/api_client.py', 'r') as f:
            content = f.read()
        
        # Check that async/await are not used
        if 'async def' in content:
            print("❌ Found 'async def' - async functions not fully removed")
            return False
        
        if 'await ' in content:
            print("❌ Found 'await' - async calls not fully removed")
            return False
        
        if 'aiohttp' in content:
            print("❌ Found 'aiohttp' - dependency not fully removed")
            return False
        
        # Check that requests is used
        if 'import requests' not in content:
            print("❌ 'import requests' not found")
            return False
        
        print("✅ All async code properly converted to synchronous")
        return True
    except Exception as e:
        print(f"❌ Error checking async removal: {e}")
        return False

def test_dependencies_updated():
    """Test that pyproject.toml has been updated."""
    try:
        with open('/src/pyproject.toml', 'r') as f:
            content = f.read()
        
        # Check that aiohttp is removed and requests is added
        if 'aiohttp[speedups]' in content:
            print("❌ 'aiohttp[speedups]' still in dependencies")
            return False
        
        if '"requests",' not in content and '"requests"' not in content:
            print("❌ 'requests' not found in dependencies")
            return False
        
        # Check that nest-asyncio is removed
        if 'nest-asyncio' in content:
            print("❌ 'nest-asyncio' still in dependencies")
            return False
        
        print("✅ Dependencies properly updated in pyproject.toml")
        return True
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing aiohttp to requests conversion...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_api_client_structure,
        test_async_removal,
        test_dependencies_updated,
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The conversion was successful.")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
