import os
import subprocess
from pathlib import Path

from google.cloud import storage


BUCKET_NAME = "shirt-fabric-models"
BASE_DIR = Path("/app")
LOCAL_MODELS_DIR = BASE_DIR / "ai_models"


def download_models():
    """
    Download AI model files from Google Cloud Storage while preserving
    the ai_models directory structure expected by the existing code.
    """

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = client.list_blobs(bucket, prefix="ai_models/")

    downloaded_files = 0

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = Path(blob.name)
        local_path = BASE_DIR / relative_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size == blob.size:
            print(f"Model file already exists: {local_path}")
            continue

        print(f"Downloading: gs://{BUCKET_NAME}/{blob.name}")

        blob.download_to_filename(str(local_path))

        downloaded_files += 1

    print(
        f"Model preparation completed. "
        f"Newly downloaded files: {downloaded_files}"
    )


if __name__ == "__main__":
    print("Preparing AI models...")

    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    download_models()

    print("Starting Django application...")

    subprocess.run(
        [
            "gunicorn",
            "config.wsgi:application",
            "--bind",
            f"0.0.0.0:{os.environ.get('PORT', '8080')}",
            "--workers",
            "1",
            "--timeout",
            "300",
            "--access-logfile",
            "-",
            "--error-logfile",
            "-",
            "--capture-output",
        ],
        check=True,
    )