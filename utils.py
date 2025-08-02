import os

def verify_bearer_token(auth_header: str) -> bool:
    """Verify Bearer token from Authorization header."""
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    # For hackathon, use the provided token
    required_token = os.getenv("HACKRX_API_KEY") or "bfb8fabaf1ce137c1402366fb3d5a052836234c1ff376c326842f52e3164cc33"
    return token == required_token
