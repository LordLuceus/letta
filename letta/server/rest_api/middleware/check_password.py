import base64

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class CheckPasswordMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, password: str):
        super().__init__(app)
        self.password = password

    def _extract_basic_auth_password(self, header_value: str) -> str | None:
        """Extract the password from a Basic auth header (ignores username)."""
        if not header_value.startswith("Basic "):
            return None
        try:
            decoded = base64.b64decode(header_value[6:]).decode("utf-8")
            # Format: username:password — we only care about the password
            _, _, password = decoded.partition(":")
            return password or None
        except Exception:
            return None

    async def dispatch(self, request, call_next):
        # Exclude health check endpoint from password protection
        if request.url.path in {"/v1/health", "/v1/health/", "/latest/health/"}:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")

        if (
            request.headers.get("X-BARE-PASSWORD") == f"password {self.password}"
            or auth_header == f"Bearer {self.password}"
            or self._extract_basic_auth_password(auth_header) == self.password
        ):
            return await call_next(request)

        # Include WWW-Authenticate header so git clients know to send
        # Basic credentials on retry (git sends unauthenticated discovery
        # requests first and relies on a 401 + WWW-Authenticate challenge).
        return JSONResponse(
            content={"detail": "Unauthorized"},
            status_code=401,
            headers={"WWW-Authenticate": "Basic realm=\"letta\""},
        )
