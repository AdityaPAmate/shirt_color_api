# api/views.py
import os
import uuid
from pathlib import Path

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse

from api.ai.pipeline_singleton import get_pipeline

# api/views.py -> .parent = api/ -> .parent.parent = project root (shirt-color-api/)
BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "media" / "uploads"
OUTPUT_DIR = BASE_DIR / "media" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ReplaceFabricView(APIView):
    """
    POST /api/replace-fabric/
        multipart/form-data:
            person_image  : file (required)
            fabric_image  : file (required)
            garment_type  : "shirt" | "kurta"  (optional, default "shirt")

        Response: image/jpeg (final rendered image)
    """

    parser_classes = [MultiPartParser]

    def post(self, request):
        person_file = request.FILES.get("person_image")
        fabric_file = request.FILES.get("fabric_image")
        garment_type = request.data.get("garment_type", "shirt")
        detection_target = request.data.get("detection_target", garment_type)

        if person_file is None or fabric_file is None:
            return Response(
                {"error": "person_image and fabric_image both files required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if garment_type not in ("shirt", "kurta"):
            return Response(
                {"error": "garment_type is any of the both 'shirt' or 'kurta'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ShirtPipeline.replace_fabric() FILE PATHS ghet -- image arrays nahi
        # (te swataha cv2.imread/cv2.imwrite karta). Tyamule uploaded files
        # aadhi disk var save karava lagtat.
        #
        # request_id (uuid) vaparto jenekarun 2 concurrent requests eka
        # doghanchya person/fabric/output files overwrite karणar nahit.
        request_id = uuid.uuid4().hex

        person_ext = Path(person_file.name).suffix or ".jpg"
        fabric_ext = Path(fabric_file.name).suffix or ".jpg"

        person_path = UPLOAD_DIR / f"{request_id}_person{person_ext}"
        fabric_path = UPLOAD_DIR / f"{request_id}_fabric{fabric_ext}"
        output_path = OUTPUT_DIR / f"{request_id}_output.jpg"

        with open(person_path, "wb") as f:
            for chunk in person_file.chunks():
                f.write(chunk)

        with open(fabric_path, "wb") as f:
            for chunk in fabric_file.chunks():
                f.write(chunk)

        # === Ithe get_pipeline() call karto -- server start hotanach
        # loaded zalela instance reuse hoto, per-request reload NAHI ===
        pipeline = get_pipeline()

        try:
            pipeline.replace_fabric(
                person_image_path=str(person_path),
                fabric_image_path=str(fabric_path),
                output_path=str(output_path),
                garment_type=garment_type,
                detection_target=detection_target,
            )
        except Exception as e:
            return Response(
                {"error": f"Pipeline fail zala: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            # temp uploaded INPUT files clean karto (output matra
            # response madhun user la pathवaycha aahe, tyamule to thevla)
            for p in (person_path, fabric_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

        if not output_path.exists():
            return Response(
                {"error": "Pipeline output file generate zala nahi."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return FileResponse(
            open(output_path, "rb"),
            content_type="image/jpeg",
            filename=f"{request_id}_output.jpg",
        )