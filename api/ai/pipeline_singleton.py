"""
File:
    api/ai/pipeline_singleton.py

Purpose:
    Ensures ShirtPipeline (which loads GroundingDINO + SAM, expensive
    models) is created only ONCE per running server process -
    never per request.

    get_pipeline() is safe to call from any view: the first call
    loads the models and caches the instance; every call after that
    just returns the already-loaded instance instantly.
"""

from api.ai.pipeline import ShirtPipeline

_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance

    if _pipeline_instance is None:
        print("\n[pipeline_singleton] Loading AI models for the first time...")
        _pipeline_instance = ShirtPipeline()
        print("[pipeline_singleton] Models loaded. All future requests will reuse them.\n")

    return _pipeline_instance