"""
File:
    api/apps.py

Purpose:
    Triggers model loading as soon as the Django server starts,
    so the FIRST request is also fast (not just the second onwards).
"""

import os
from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        # ------------------------------------------------------------
        # Django च्या runserver auto-reloader मध्ये ready() दोनदा
        # चालू शकतो (एक "watcher" process, एक खरा "worker" process).
        # RUN_MAIN तपासून फक्त खऱ्या worker process मध्येच models
        # load करतो -> डबल-लोडिंग टाळतो.
        # ------------------------------------------------------------
        if os.environ.get("RUN_MAIN") != "true":
            return

        from api.ai.pipeline_singleton import get_pipeline
        get_pipeline()   # <- इथेच, server सुरू होताच, models load होतात