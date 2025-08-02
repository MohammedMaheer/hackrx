import os
import logging
from typing import Dict
import hashlib
import hmac
from typing import Optional

logger = logging.getLogger(__name__)

def verify_bearer_token(auth_header: str) -> bool:
    """Verify Bearer token from Authorization header with enhanced security"""
    
    if not auth_header:
        logger.warning("No authorization header provided")
        return False
    
    # Check if header starts with "Bearer "
    if not auth_header.lower().startswith("bearer "):
        logger.warning("Authorization header doesn't start with 'Bearer '")
        return False
    
    # Extract token
    try:
        token = auth_header.split(" ", 1)[1].strip()
    except IndexError:
        logger.warning("No token found after 'Bearer '")
        return False
    
    if not token:
        logger.warning("Empty token provided")
        return False
    
    # Get required token from environment
    required_token = os.getenv("HACKRX_API_KEY")
    
    if not required_token:
        # Fallback to hardcoded token for development
        required_token = "bfb8fabaf1ce137c1402366fb3d5a052836234c1ff376c326842f52e3164cc33"
        logger.warning("Using fallback token - set HACKRX_API_KEY environment variable")
    
    # Use constant-time comparison to prevent timing attacks
    try:
        is_valid = hmac.compare_digest(token, required_token)
        if is_valid:
            logger.info("Token validation successful")
        else:
            logger.warning("Token validation failed")
        return is_valid
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return False

def validate_url(url: str) -> bool:
    """Validate if the provided URL is acceptable"""
    if not url:
        return False
    
    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Block potentially dangerous URLs
    blocked_domains = ['localhost', '127.0.0.1', '0.0.0.0', 'file://']
    for blocked in blocked_domains:
        if blocked in url.lower():
            logger.warning(f"Blocked URL contains restricted domain: {blocked}")
            return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    if not filename:
        return "document"
    
    # Remove path separators and dangerous characters
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\.{2,}', '.', sanitized)  # Remove multiple dots
    
    # Limit length
    if len(sanitized) > 100:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:95] + ext
    
    return sanitized or "document"

def get_file_hash(file_path: str) -> Optional[str]:
    """Generate SHA-256 hash of file for caching/deduplication"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to generate file hash: {e}")
        return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are present"""
    keys_status = {
        "HACKRX_API_KEY": bool(os.getenv("HACKRX_API_KEY")),
        "PERPLEXITY_API_KEY": bool(os.getenv("PERPLEXITY_API_KEY")),
        "COHERE_API_KEY": bool(os.getenv("COHERE_API_KEY")),
        "PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY"))
    }
    
    missing_keys = [key for key, present in keys_status.items() if not present]
    
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
    else:
        logger.info("All API keys are present")
    
    return keys_status

def log_request_info(request_data: dict, user_ip: str = None):
    """Log request information for monitoring and debugging"""
    try:
        doc_url = request_data.get("documents", "")
        questions_count = len(request_data.get("questions", []))
        
        logger.info(f"Request - Doc: {doc_url[:100]}..., Questions: {questions_count}, IP: {user_ip}")
        
        # Log question types for analytics
        questions = request_data.get("questions", [])
        for i, q in enumerate(questions[:5]):  # Log first 5 questions
            logger.info(f"Q{i+1}: {q[:100]}...")
            
    except Exception as e:
        logger.error(f"Failed to log request info: {e}")

def extract_domain(url: str) -> str:
    """Extract domain from URL for logging/analytics"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return "unknown"

class PerformanceMonitor:
    """Simple performance monitoring for the API"""
    
    def __init__(self):
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        self.start_time = time.time()
    
    def record_request(self, processing_time: float, error: bool = False):
        """Record request metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time
        if error:
            self.error_count += 1
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        uptime = time.time() - self.start_time
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_processing_time": avg_processing_time,
            "requests_per_minute": self.request_count / max(uptime / 60, 1)
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()