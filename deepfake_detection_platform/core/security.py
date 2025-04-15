from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
import secrets
import logging
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)

# API Key authentication
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)) -> Optional[str]:
    """
    Get and validate the API key from the request header.
    
    Args:
        api_key: API key from the request header
        
    Returns:
        Validated API key if authentication is successful
        
    Raises:
        HTTPException: If authentication fails
    """
    # If no API key is set in settings, authentication is disabled
    if not settings.API_KEY:
        return None
    
    # Otherwise, verify the provided API key
    if api_key and secrets.compare_digest(api_key, settings.API_KEY):
        return api_key
    
    logger.warning("Invalid API key provided")
    raise HTTPException(
        status_code=401,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "APIKey"}
    )

# Rate limiting middleware
class RateLimiter:
    """
    Simple in-memory rate limiting middleware for FastAPI.
    """
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.timestamps = {}
    
    async def __call__(self, request: Request):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get current time
        import time
        current_time = time.time()
        
        # Initialize or clean up old entries
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = 0
            self.timestamps[client_ip] = []
        
        # Remove timestamps older than 1 minute
        minute_ago = current_time - 60
        self.timestamps[client_ip] = [t for t in self.timestamps[client_ip] if t > minute_ago]
        
        # Check if rate limit exceeded
        if len(self.timestamps[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        self.timestamps[client_ip].append(current_time)
        self.request_counts[client_ip] += 1
        
        # Clean up old clients' data periodically
        if self.request_counts[client_ip] % 100 == 0:
            self._cleanup_old_data(current_time)
    
    def _cleanup_old_data(self, current_time: float):
        """
        Clean up data for clients that haven't made requests in the last 10 minutes.
        
        Args:
            current_time: Current timestamp
        """
        ten_minutes_ago = current_time - 600
        
        to_delete = []
        for client_ip, timestamps in self.timestamps.items():
            if not timestamps or max(timestamps) < ten_minutes_ago:
                to_delete.append(client_ip)
        
        for client_ip in to_delete:
            del self.request_counts[client_ip]
            del self.timestamps[client_ip]

# Create a rate limiter instance
rate_limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)

# Generate secure tokens
def generate_secure_token() -> str:
    """
    Generate a secure random token.
    
    Returns:
        A secure random token string
    """
    return secrets.token_urlsafe(32)
