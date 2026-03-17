"""Opik client singleton with graceful degradation.

Returns None when OPIK_URL_OVERRIDE / OPIK_API_KEY is not set or opik is not
installed, so every caller can guard with ``if client:``.
"""

import os
from contextlib import suppress

import opik
from dotenv import load_dotenv


_client = None
_initialised = False


def get_client():
    """
    Initialize and return a globally cached Opik client.

    Retrieves configuration from environment variables and configures
    the SDK for local or cloud usage.

    Returns
    -------
    Opik or None
        An active client, or None if configuration is missing or invalid.
    """
    global _client, _initialised  # noqa: PLW0603
    if _initialised:
        return _client

    _initialised = True

    # Load environment variables from .env file
    with suppress(Exception):
        load_dotenv()

    url = os.environ.get("OPIK_URL_OVERRIDE", "")
    api_key = os.environ.get("OPIK_API_KEY", "")

    if not url and not api_key:
        return None

    try:
        if url:
            # Opik SDK expects the base URL without /api suffix
            base_url = url.rstrip("/")
            if base_url.endswith("/api"):
                base_url = base_url[:-4]
            opik.configure(url=base_url, use_local=True, force=True, automatic_approvals=True)
        else:
            opik.configure(api_key=api_key, force=True, automatic_approvals=True)

        _client = opik.Opik()
    except Exception as exc:
        print(f"[opik] client init failed: {exc}")
        _client = None

    return _client


def reset_client() -> None:
    """
    Clear the cached Opik client and force re-initialization.

    Returns
    -------
    None
    """
    global _client, _initialised  # noqa: PLW0603
    _client = None
    _initialised = False
