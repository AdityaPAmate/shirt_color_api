import logging
import os
import sys

from django.apps import AppConfig


logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        """
        Preload AI models during application startup.

        Development behavior:
            Django's runserver autoreloader creates two processes:
            a parent watcher process and a child server process. The ready()
            method runs in both processes, so the RUN_MAIN guard prevents the
            heavy AI models from being loaded twice.

        Production behavior:
            Gunicorn does not use Django's runserver autoreloader. Therefore,
            the guard is skipped and the pipeline is preloaded normally.
        """
        is_runserver = "runserver" in sys.argv

        if is_runserver and os.environ.get("RUN_MAIN") != "true":
            logger.info(
                "MODEL_PRELOAD_SKIPPED | reason=runserver_autoreloader_parent"
            )
            return

        logger.info("APPLICATION_STARTUP | pipeline_preload=started")

        from api.ai.pipeline_singleton import preload_pipeline

        preload_pipeline()

        logger.info("APPLICATION_STARTUP | pipeline_preload=completed")