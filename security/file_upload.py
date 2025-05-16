#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Secure File Upload Module

Provides secure file upload functionality with content validation
and malicious content scanning for the Super AI API.
"""

import os
import uuid
import magic
import hashlib
import logging
from typing import Tuple, Dict, Any, Optional, List
from werkzeug.utils import secure_filename
from flask import current_app, request

# Configure logging
logger = logging.getLogger("file_upload")

# Allowed file extensions and their corresponding MIME types
ALLOWED_EXTENSIONS = {
    'csv': 'text/csv',
    'json': 'application/json',
    'txt': 'text/plain',
}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_content(file_path: str) -> Tuple[bool, str]:
    """
    Validate file content by checking MIME type and scanning for malicious content

    Args:
        file_path: Path to the uploaded file

    Returns:
        Tuple of (is_valid, message)
    """
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        return False, f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB"

    # Check MIME type
    mime = magic.Magic(mime=True)
    file_mime = mime.from_file(file_path)

    # Get file extension from path
    file_ext = os.path.splitext(file_path)[1][1:].lower()

    # Verify MIME type matches expected type for the extension
    expected_mime = ALLOWED_EXTENSIONS.get(file_ext)
    if expected_mime and file_mime != expected_mime:
        return False, f"File MIME type ({file_mime}) does not match expected type for {file_ext} file"

    # Check for executable content or scripts
    if file_mime in ['application/x-executable', 'application/x-dosexec', 'application/x-sharedlib']:
        return False, "Executable files are not allowed"

    # Simple content scan (this should be enhanced with proper malware scanning in production)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(8192)  # Read first 8KB for scanning

            # Check for potentially malicious patterns
            suspicious_patterns = [
                '<script', 'eval(', 'document.cookie', 'exec(',
                'system(', 'os.system', 'subprocess', 'ProcessBuilder'
            ]

            for pattern in suspicious_patterns:
                if pattern.lower() in content.lower():
                    return False, f"Suspicious content detected: {pattern}"

    except Exception as e:
        logger.error(f"Error scanning file content: {str(e)}")
        return False, "Error scanning file content"

    return True, "File is valid"

def save_uploaded_file(file, upload_dir: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Save an uploaded file securely with validation

    Args:
        file: The file object from request.files
        upload_dir: Directory to save the file
        metadata: Optional metadata to store with the file

    Returns:
        Dict with file information or error
    """
    # Create upload directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)

    # Check if the file was uploaded
    if not file:
        return {"success": False, "error": "No file provided"}

    # Check if the filename is secure
    if file.filename == '':
        return {"success": False, "error": "No file selected"}

    filename = secure_filename(file.filename)

    # Check if the file type is allowed
    if not allowed_file(filename):
        return {
            "success": False,
            "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS.keys())}"
        }

    # Generate a unique filename to prevent overwriting
    file_extension = os.path.splitext(filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)

    try:
        # Save the file
        file.save(file_path)

        # Calculate file hash for integrity checking
        file_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        file_hash_str = file_hash.hexdigest()

        # Validate file content
        is_valid, validation_message = validate_file_content(file_path)

        if not is_valid:
            # If validation fails, delete the file and return error
            os.remove(file_path)
            return {"success": False, "error": validation_message}

        # Create response with file information
        result = {
            "success": True,
            "filename": unique_filename,
            "original_filename": filename,
            "size": os.path.getsize(file_path),
            "hash": file_hash_str,
            "upload_time": os.path.getctime(file_path),
        }

        # Add metadata if provided
        if metadata:
            result["metadata"] = metadata

        return result

    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        # Clean up if file was saved
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"success": False, "error": f"Error saving file: {str(e)}"}

# Function to register upload route with Flask app
def register_upload_routes(app, jwt_required_func):
    """Register secure file upload routes with the Flask app"""

    @app.route("/api/upload", methods=["POST"])
    @jwt_required_func()
    def upload_file():
        """Secure file upload endpoint"""
        # Get upload directory from config or use default
        upload_dir = app.config.get("UPLOAD_DIR", "data/uploads")

        # Check if a file was uploaded
        if 'file' not in request.files:
            return {"error": "No file part in the request"}, 400

        file = request.files['file']

        # Get metadata from form data if available
        metadata = {}
        if request.form.get('metadata'):
            try:
                import json
                metadata = json.loads(request.form.get('metadata'))
            except:
                return {"error": "Invalid metadata format"}, 400

        # Save the file
        result = save_uploaded_file(file, upload_dir, metadata)

        if not result["success"]:
            return {"error": result["error"]}, 400

        return result, 200