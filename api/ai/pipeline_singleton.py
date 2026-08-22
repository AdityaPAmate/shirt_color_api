# api/ai/pipeline_singleton.py
"""
ShirtPipeline chi ek global singleton instance provide karnyasathi ha module.

Ka lagto:
    ShirtPipeline.__init__() madhe GroundingDINO + SAM 2.1 load hotat --
    he heavy, slow (CPU-only) operations aahet. Pratek API request la
    navin ShirtPipeline() banवला tar pratek request 10-30+ sec lagतील.

    Ha module ekach global variable (_pipeline_instance) madhe pipeline
    cache karto. AppConfig.ready() server start hotana ha eager-load
    karto; test_pipeline.py sudhha hach function vaparto, tyamule
    Django server ANI standalone test script doghehi shared/consistent
    loading logic vapartat -- duplicate code nahi.
"""

from api.ai.pipeline import ShirtPipeline

_pipeline_instance = None


def get_pipeline() -> ShirtPipeline:
    """
    Global singleton accessor.

    Pahilyanda call zalyavar ShirtPipeline() banto (GroundingDINO + SAM2.1
    load -- slow, EKDACH hoto). Tyanantar pudhchya pratek call la
    tich cached instance return hoto (fast, model reload nahi).
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        print("[pipeline_singleton] Loading ShirtPipeline (GroundingDINO + SAM2.1)... ekdach hoil.")
        _pipeline_instance = ShirtPipeline()
        print("[pipeline_singleton] ShirtPipeline loaded and cached in memory.")
    return _pipeline_instance


def preload_pipeline():
    """AppConfig.ready() madhun explicitly call karnyasathi -- eager loading trigger."""
    get_pipeline()