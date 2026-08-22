# api/apps.py
import os
import sys
from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        """
        Server start hotana (fakt EKDA) AI models load karnyasathi.

        RUN_MAIN guard ka lagto:
            `python manage.py runserver` vaparल्यावर Django cha
            autoreloader EK parent (watcher) process + EK child process
            asे 2 process spawn karto. ready() dohi process madhe call
            hoto. Guard nasel tar GroundingDINO + SAM2.1 DONDA load
            hotil (double memory, double load-time).

            Child process madhech RUN_MAIN="true" environment variable
            set aste -- tyamule tithech actual loading karto.

        Production (gunicorn/uwsgi) madhe `runserver` argv madhe naste,
        tyamule ha guard skip hoto ani ready() normally ekdach chalto.
        """
        is_runserver = "runserver" in sys.argv

        if is_runserver and os.environ.get("RUN_MAIN") != "true":
            # ha autoreloader cha parent (watcher) process aahe -- ithe
            # load karaycha nahi, child process madhech khara loading hoil.
            return

        from api.ai.pipeline_singleton import preload_pipeline
        preload_pipeline()