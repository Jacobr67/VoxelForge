"""
src/api/trellis_client.py

Handles all communication with the NVIDIA Trellis NIM hosted endpoint.
Sends text prompts, polls for job completion, and returns the raw GLB
mesh bytes ready for the voxeliser pipeline.

NVIDIA Trellis NIM endpoint:
  POST https://ai.api.nvidia.com/v1/genai/microsoft/trellis
  Auth: Authorization: Bearer <NGC_API_KEY>

Actual response format:
  {"artifacts": [{"base64": "Z2xURg..."}]}
"""

import base64
import time
import logging
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

TRELLIS_ENDPOINT  = "https://ai.api.nvidia.com/v1/genai/microsoft/trellis"
DEFAULT_TIMEOUT   = 30
POLL_INTERVAL     = 3
MAX_POLL_ATTEMPTS = 100


# ── Exceptions ─────────────────────────────────────────────────────────────────

class TrellisAuthError(Exception):
    """Raised when the API key is missing or rejected."""

class TrellisRequestError(Exception):
    """Raised when the API returns an unexpected error response."""

class TrellisTimeoutError(Exception):
    """Raised when a generation job exceeds the maximum wait time."""


# ── Client ─────────────────────────────────────────────────────────────────────

class TrellisClient:
    """
    Thin wrapper around the NVIDIA Trellis NIM REST API.

    Usage:
        client = TrellisClient(api_key="nvapi-...")
        glb_bytes = client.generate_from_text("a medieval stone castle")

    The returned bytes are a binary GLB file that can be passed directly
    to trimesh.load() in the voxeliser pipeline.
    """

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise TrellisAuthError(
                "No API key provided. Set NVIDIA_API_KEY in your .env file."
            )
        self._api_key = api_key.strip()
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Accept":        "application/json",
            "Content-Type":  "application/json",
        })

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_from_text(
        self,
        prompt: str,
        seed: int = 0,
        on_progress=None,
        **kwargs,   # absorb unused settings args silently
    ) -> bytes:
        """
        Generate a 3D mesh from a text prompt and return raw GLB bytes.

        Args:
            prompt:      Natural language description of the structure.
            seed:        Reproducibility seed. 0 = random.
            on_progress: Optional callback(status: str) for UI updates.

        Returns:
            Raw GLB file bytes ready for trimesh.load(io.BytesIO(glb_bytes)).

        Raises:
            TrellisAuthError:    Invalid or missing API key.
            TrellisRequestError: API returned an error.
            TrellisTimeoutError: Job did not complete in time.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        logger.info(f"Submitting Trellis job: '{prompt[:60]}...'")
        self._notify(on_progress, "Submitting generation request...")

        payload = {"prompt": prompt.strip()}
        if seed and seed != 0:
            payload["seed"] = seed

        response = self._post(TRELLIS_ENDPOINT, payload)

        if response.status_code == 200:
            logger.info("Received synchronous response.")
            self._notify(on_progress, "Mesh received. Processing...")
            return self._extract_glb(response)

        if response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            if not request_id:
                raise TrellisRequestError(
                    "Server returned 202 but no NVCF-REQID header was found."
                )
            logger.info(f"Async job queued. Request ID: {request_id}")
            return self._poll_for_result(request_id, on_progress)

        self._raise_for_status(response)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _poll_for_result(self, request_id: str, on_progress) -> bytes:
        """Poll the NVCF status endpoint until the job is done."""
        status_url = (
            f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{request_id}"
        )
        for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
            self._notify(on_progress, f"Generating mesh... ({attempt * POLL_INTERVAL}s)")
            time.sleep(POLL_INTERVAL)

            resp = self._session.get(status_url, timeout=DEFAULT_TIMEOUT)

            if resp.status_code == 200:
                logger.info(f"Job complete after {attempt} polls.")
                self._notify(on_progress, "Mesh received. Processing...")
                return self._extract_glb(resp)

            if resp.status_code == 202:
                logger.debug(f"Still processing (poll {attempt}).")
                continue

            if resp.status_code == 401:
                raise TrellisAuthError("API key rejected during polling.")

            self._raise_for_status(resp)

        raise TrellisTimeoutError(
            f"Trellis job did not complete after "
            f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL} seconds."
        )

    def _extract_glb(self, response: requests.Response) -> bytes:
        """
        Extract raw GLB bytes from the API response.

        NVIDIA Trellis returns:
            {"artifacts": [{"base64": "Z2xURg..."}]}

        Falls back to handling raw binary and URL-based responses too.
        """
        content_type = response.headers.get("Content-Type", "")

        # ── Format 1: raw binary body ─────────────────────────────────────────
        if "model/gltf-binary" in content_type or "application/octet-stream" in content_type:
            logger.info("GLB received as raw binary body.")
            return response.content

        # ── Format 2: JSON envelope ───────────────────────────────────────────
        if "application/json" in content_type:
            data = response.json()
            logger.debug(f"Trellis JSON response keys: {list(data.keys())}")

            # Primary format: {"artifacts": [{"base64": "..."}]}
            if "artifacts" in data and isinstance(data["artifacts"], list):
                artifact = data["artifacts"][0]
                logger.debug(f"Artifact keys: {list(artifact.keys())}")

                if "base64" in artifact:
                    logger.info("Decoding GLB from artifacts[0].base64")
                    return base64.b64decode(artifact["base64"])

                if "url" in artifact:
                    logger.info(f"Fetching GLB from artifacts[0].url")
                    asset_resp = self._session.get(artifact["url"], timeout=120)
                    asset_resp.raise_for_status()
                    return asset_resp.content

            # Fallback: top-level "glb" key (base64)
            if "glb" in data:
                logger.info("Decoding GLB from top-level 'glb' key.")
                return base64.b64decode(data["glb"])

            # Fallback: top-level URL key
            if "glb_url" in data:
                logger.info("Fetching GLB from top-level 'glb_url'.")
                asset_resp = self._session.get(data["glb_url"], timeout=120)
                asset_resp.raise_for_status()
                return asset_resp.content

            # Nothing matched — log full response for future diagnosis
            logger.error(
                f"Could not extract GLB from JSON response.\n"
                f"Full response (first 2000 chars):\n{str(data)[:2000]}"
            )

        raise TrellisRequestError(
            f"Unrecognised response format from Trellis API. "
            f"Content-Type: {content_type}. "
            f"Check logs/voxelforge.log for the full response."
        )

    def _post(self, url: str, payload: dict) -> requests.Response:
        """Send a POST request, raising auth errors immediately."""
        try:
            response = self._session.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        except requests.ConnectionError as e:
            raise TrellisRequestError(
                f"Could not connect to NVIDIA API. Check your internet connection.\n{e}"
            )
        except requests.Timeout:
            raise TrellisRequestError(
                "Request to NVIDIA API timed out. Try again."
            )
        if response.status_code == 401:
            raise TrellisAuthError(
                "NVIDIA API key is invalid or expired. "
                "Update NVIDIA_API_KEY in your .env file."
            )
        return response

    def _raise_for_status(self, response: requests.Response):
        """Raise a descriptive TrellisRequestError for non-2xx responses."""
        try:
            detail = response.json().get("detail", response.text[:200])
        except Exception:
            detail = response.text[:200]
        raise TrellisRequestError(
            f"Trellis API error {response.status_code}: {detail}"
        )

    @staticmethod
    def _notify(callback, message: str):
        """Fire the optional progress callback safely."""
        if callable(callback):
            try:
                callback(message)
            except Exception:
                pass