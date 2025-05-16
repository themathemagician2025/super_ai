# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import asyncio
import aiohttp
import sys
from pathlib import Path
import json

async def test_api_endpoints():
    """Test the API endpoints."""
    async with aiohttp.ClientSession() as session:
        # Test home endpoint
        async with session.get('http://127.0.0.1:5000/') as response:
            assert response.status == 200
            data = await response.json()
            assert data['message'] == "Welcome to Super AI API"
            print("✓ Home endpoint test passed")

        # Test health endpoint
        async with session.get('http://127.0.0.1:5000/health') as response:
            assert response.status == 200
            data = await response.json()
            assert data['status'] == "healthy"
            print("✓ Health endpoint test passed")

        # Test metrics endpoint
        async with session.get('http://127.0.0.1:5000/metrics') as response:
            assert response.status == 200
            data = await response.json()
            assert 'timestamp' in data
            assert 'system_status' in data
            print("✓ Metrics endpoint test passed")

async def main():
    """Run the system tests."""
    print("\nRunning system tests...")
    try:
        await test_api_endpoints()
        print("\nAll tests passed! ✓")
    except AssertionError as e:
        print(f"\nTest failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during tests: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
