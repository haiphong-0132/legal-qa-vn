"""
API client module để giao tiếp với remote FastAPI server
hosting embedding, rerank, và generate models
"""

from .remote_client import APIConfig, RemoteAPIClient

__all__ = [
    "RemoteAPIClient",
    "APIConfig",
]
