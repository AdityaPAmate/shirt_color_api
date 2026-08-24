"""
Provides a global singleton instance of ShirtPipeline.

Why this is needed:
    ShirtPipeline.__init__() loads GroundingDINO and SAM 2.1. These are
    heavy and slow CPU-only operations. Creating a new ShirtPipeline instance
    for every API request would reload the models and significantly increase
    request processing time.

    This module stores one global pipeline instance in _pipeline_instance.
    AppConfig.ready() can preload it when the server starts. Standalone scripts
    can also use the same accessor, keeping model-loading logic centralized.
"""

import logging

from api.ai.pipeline import ShirtPipeline


logger = logging.getLogger(__name__)

_pipeline_instance = None


def get_pipeline() -> ShirtPipeline:
    """
    Return the global ShirtPipeline singleton instance.

    On the first call, ShirtPipeline is created and the AI models are loaded.
    Subsequent calls return the cached instance without reloading the models.
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        logger.info(
            "MODEL_LOADING_STARTED | models=GroundingDINO,SAM2.1"
        )

        _pipeline_instance = ShirtPipeline()

        logger.info(
            "MODEL_LOADING_COMPLETED | pipeline=ShirtPipeline"
        )

    return _pipeline_instance


def preload_pipeline():
    """
    Explicitly trigger eager loading of the global pipeline instance.
    Called from AppConfig.ready() during application startup.
    """
    logger.info("PIPELINE_PRELOAD_STARTED")

    get_pipeline()

    logger.info("PIPELINE_PRELOAD_COMPLETED")