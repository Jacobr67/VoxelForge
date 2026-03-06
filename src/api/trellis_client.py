"""
src/api/trellis_client.py

Handles all communication with the NVIDIA Trellis NIM hosted endpoint.
Sends text prompts, polls for job completion, and returns the raw GLB
mesh bytes ready for the voxeliser pipeline.

NVIDIA Trellis NIM endpoint:
  POST https://ai.api.nvidia.com/v1/genai/microsoft/trellis
  Auth: Authorization: Bearer <NGC_API_KEY>
"""

import io
import time
import logging
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

TRELLIS_ENDPOINT = "https://ai.api.nvidia.com/v1/genai/microsoft/trellis"
ASSETS_ENDPOINT  = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

DEFAULT_TIMEOUT      = 30    # seconds per HTTP request
POLL_INTERVAL        = 3     # seconds between status polls
MAX_POLL_ATTEMPTS    = 100   # ~5 minutes before giving up


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
        """
        Args:
            api_key: Your NVIDIA NGC API key (starts with 'nvapi-').
                     Store this in config/api_keys.yaml (never commit it)
                     or in a .env file as NVIDIA_API_KEY=nvapi-...
        """
        if not api_key or not api_key.strip():
            raise TrellisAuthError(
                "No API key provided. Set NVIDIA_API_KEY in your .env file "
                "or in config/api_keys.yaml."
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
        texture_resolution: int = 1024,
        geometry_fidelity:  float = 0.5,
        surface_fidelity:   float = 0.5,
        seed:               int = 0,
        on_progress=None,
    ) -> bytes:
        """
        Generate a 3D mesh from a text prompt and return it as GLB bytes.

        Args:
            prompt:              Natural language description of the structure.
            texture_resolution:  Output texture size in pixels (512, 1024, 2048).
            geometry_fidelity:   0.0–1.0. Higher = more geometric detail.
            surface_fidelity:    0.0–1.0. Higher = more surface detail.
            seed:                Reproducibility seed. 0 = random.
            on_progress:         Optional callback(status: str) for UI updates.

        Returns:
            Raw GLB file bytes. Pass directly to trimesh.load(io.BytesIO(glb_bytes)).

        Raises:
            TrellisAuthError:    Invalid or missing API key.
            TrellisRequestError: API returned an error.
            TrellisTimeoutError: Job did not complete in time.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        logger.info(f"Submitting Trellis job: '{prompt[:60]}...'")
        self._notify(on_progress, "Submitting generation request...")

        # ── 1. Submit the generation job ──────────────────────────────────────
        payload = {
            "text_prompts": [{"text": prompt.strip()}],
            "texture_resolution": texture_resolution,
            "geometry_fidelity":  geometry_fidelity,
            "surface_fidelity":   surface_fidelity,
            "seed": seed,
        }

        response = self._post(TRELLIS_ENDPOINT, payload)

        # ── 2. Handle synchronous vs async response ───────────────────────────
        # Trellis NIM returns 200 with GLB directly, or 202 with a polling URL.
        if response.status_code == 200:
            logger.info("Received synchronous GLB response.")
            self._notify(on_progress, "Mesh received.")
            return self._extract_glb(response)

        if response.status_code == 202:
            # Async job — poll until complete
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
        status_url = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{request_id}"

        for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
            self._notify(
                on_progress,
                f"Generating mesh... ({attempt * POLL_INTERVAL}s)"
            )
            time.sleep(POLL_INTERVAL)

            resp = self._session.get(status_url, timeout=DEFAULT_TIMEOUT)

            if resp.status_code == 200:
                logger.info(f"Job {request_id} complete after {attempt} polls.")
                self._notify(on_progress, "Mesh received.")
                return self._extract_glb(resp)

            if resp.status_code == 202:
                # Still processing — keep polling
                logger.debug(f"Job {request_id} still processing (poll {attempt}).")
                continue

            if resp.status_code == 401:
                raise TrellisAuthError("API key rejected during polling.")

            self._raise_for_status(resp)

        raise TrellisTimeoutError(
            f"Trellis job {request_id} did not complete after "
            f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL} seconds."
        )

    def _extract_glb(self, response: requests.Response) -> bytes:
        """
        Extract raw GLB bytes from the API response.
        The NIM endpoint may return binary GLB directly or JSON with a URL.
        """
        content_type = response.headers.get("Content-Type", "")

        # Direct binary GLB in response body
        if "model/gltf-binary" in content_type or "application/octet-stream" in content_type:
            return response.content

        # JSON response containing a download URL or base64 asset
        if "application/json" in content_type:
            data = response.json()

            # URL to download the asset separately
            if "glb_url" in data:
                logger.info("Fetching GLB from asset URL...")
                asset_resp = self._session.get(data["glb_url"], timeout=60)
                asset_resp.raise_for_status()
                return asset_resp.content

            # Direct base64-encoded GLB
            if "glb" in data:
                import base64
                return base64.b64decode(data["glb"])

        raise TrellisRequestError(
            f"Unrecognised response format from Trellis API. "
            f"Content-Type: {content_type}"
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
                pass  # Never let a UI callback crash the API thread