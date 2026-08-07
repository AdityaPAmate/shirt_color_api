"""
File:
    api/views.py

Purpose:
    HTTP endpoint that wraps ShirtPipeline.replace_fabric().
    Uses get_pipeline() so models are NEVER re-loaded per request.
"""

import os
import uuid
from django.conf import settings
from django.http import FileResponse, JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser

from api.ai.pipeline_singleton import get_pipeline


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def replace_fabric_view(request):
    """
    POST /api/replace-fabric/

    multipart/form-data:
        person_image : person photo (file)
        fabric_image  : uploaded fabric swatch (file)

    Returns the rendered output image (image/jpeg).
    """

    # ----------------------------------------------------------
    # NOTE: get_pipeline() इथे कॉल होतो, पण तो नवीन model load
    # करत नाही -> apps.py च्या ready() मध्ये आधीच load झालेला
    # instance फक्त परत मिळतो (जवळपास instant).
    # ----------------------------------------------------------
    pipeline = get_pipeline()

    person_file = request.FILES.get("person_image")
    fabric_file = request.FILES.get("fabric_image")

    if person_file is None or fabric_file is None:
        return JsonResponse(
            {"error": "Both 'person_image' and 'fabric_image' files are required."},
            status=400
        )

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    output_dir = os.path.join(settings.MEDIA_ROOT, "outputs")

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    request_id = uuid.uuid4().hex

    person_path = os.path.join(upload_dir, f"{request_id}_person.jpg")
    fabric_path = os.path.join(upload_dir, f"{request_id}_fabric.jpg")
    output_path = os.path.join(output_dir, f"{request_id}_output.jpg")

    with open(person_path, "wb") as f:
        for chunk in person_file.chunks():
            f.write(chunk)

    with open(fabric_path, "wb") as f:
        for chunk in fabric_file.chunks():
            f.write(chunk)

    try:
        pipeline.replace_fabric(
            person_image_path=person_path,
            fabric_image_path=fabric_path,
            output_path=output_path,
            fabric_mode="tile"
        )
    except Exception as e:
        return JsonResponse(
            {"error": f"Pipeline failed: {str(e)}"},
            status=500
        )

    return FileResponse(
        open(output_path, "rb"),
        content_type="image/jpeg"
    )