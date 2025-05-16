# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import sys
import os
import logging
import traceback
from typing import Any, Dict, List, Optional

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_zero_division():
    try:
        result = 1 / 0
    except Exception as e:
        logger.error(f"Zero division error: {str(e)}")
        logger.debug(f"Exception traceback: {traceback.format_exc()}")
        return False
    return True

def test_index_error():
    try:
        my_list = [1, 2, 3]
        item = my_list[10]  # This will cause an IndexError
    except Exception as e:
        logger.error(f"Index error: {str(e)}")
        logger.debug(f"Exception traceback: {traceback.format_exc()}")
        return False
    return True

def test_type_error():
    try:
        result = "string" + 5  # This will cause a TypeError
    except Exception as e:
        logger.error(f"Type error: {str(e)}")
        logger.debug(f"Exception traceback: {traceback.format_exc()}")
        return False
    return True

def test_file_not_found():
    try:
        with open("nonexistent_file.txt", "r") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"File error: {str(e)}")
        logger.debug(f"Exception traceback: {traceback.format_exc()}")
        return False
    return True

def main():
    tests = [
        ("Zero Division Test", test_zero_division),
        ("Index Error Test", test_index_error),
        ("Type Error Test", test_type_error),
        ("File Not Found Test", test_file_not_found)
    ]

    success = True
    for name, test_func in tests:
        logger.info(f"Running: {name}")
        if test_func():
            logger.info(f"{name} passed (should have failed)")
            success = False
        else:
            logger.info(f"{name} caught exception as expected")

    if success:
        logger.info("All tests completed successfully - exceptions were properly caught and handled")
        return 0
    else:
        logger.error("One or more tests failed to catch exceptions")
        return 1

if __name__ == "__main__":
    sys.exit(main())
