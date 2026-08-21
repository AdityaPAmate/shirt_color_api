"""
File:
    test_media_pose.py  (exploratory/test script — not yet part of the pipeline)

Purpose:
    Run MediaPipe Pose Landmarker on a sample person image and save:
    1. All detected pose landmarks as a text file
    2. A JSON file containing landmark coordinates
    3. A visualized PNG showing the detected pose
"""

from pathlib import Path
import json

import cv2
import mediapipe as mp


# ----------------------------------------------------------
# Input / Output paths
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

PERSON_IMAGE = f"{BASE_DIR}/test_images/person4.jpeg"

POSE_OUTPUT_DIR = f"{BASE_DIR}/test_images/media_pose"

Path(POSE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# MediaPipe Pose Landmarker
# ----------------------------------------------------------
def run_media_pose(person_image_path: str, output_dir: str):

    # ------------------------------------------------------
    # 1. MediaPipe Pose model path
    # ------------------------------------------------------
    MODEL_PATH = f"{BASE_DIR}/ai_models/mediapipe/pose_landmarker_full.task"

    # ------------------------------------------------------
    # 2. Create Pose Landmarker
    # ------------------------------------------------------
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    # ------------------------------------------------------
    # 3. Read person image
    # ------------------------------------------------------
    image = mp.Image.create_from_file(person_image_path)

    # ------------------------------------------------------
    # 4. Run pose detection
    # ------------------------------------------------------
    with PoseLandmarker.create_from_options(options) as landmarker:

        result = landmarker.detect(image)

    # ------------------------------------------------------
    # 5. Check result
    # ------------------------------------------------------
    if not result.pose_landmarks:

        print("No human pose detected.")

        return {}

    # First detected person
    landmarks = result.pose_landmarks[0]

    # ------------------------------------------------------
    # 6. Landmark names
    # ------------------------------------------------------
    landmark_names = [
        "NOSE",
        "LEFT_EYE_INNER",
        "LEFT_EYE",
        "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER",
        "RIGHT_EYE",
        "RIGHT_EYE_OUTER",
        "LEFT_EAR",
        "RIGHT_EAR",
        "MOUTH_LEFT",
        "MOUTH_RIGHT",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_PINKY",
        "RIGHT_PINKY",
        "LEFT_INDEX",
        "RIGHT_INDEX",
        "LEFT_THUMB",
        "RIGHT_THUMB",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    ]

    # ------------------------------------------------------
    # 7. Image dimensions
    # ------------------------------------------------------
    original_image = cv2.imread(person_image_path)

    height, width = original_image.shape[:2]

    # ------------------------------------------------------
    # 8. Save landmark coordinates
    # ------------------------------------------------------
    landmarks_data = {}

    for index, landmark in enumerate(landmarks):

        name = landmark_names[index]

        x_pixel = int(landmark.x * width)
        y_pixel = int(landmark.y * height)

        landmarks_data[name] = {
            "index": index,
            "x_normalized": float(landmark.x),
            "y_normalized": float(landmark.y),
            "z": float(landmark.z),
            "x_pixel": x_pixel,
            "y_pixel": y_pixel,
            "visibility": float(landmark.visibility),
        }

        print(
            f"{name:20s} "
            f"x={x_pixel:4d}, "
            f"y={y_pixel:4d}, "
            f"visibility={landmark.visibility:.3f}"
        )

    # ------------------------------------------------------
    # 9. Save JSON
    # ------------------------------------------------------
    json_path = f"{output_dir}/pose_landmarks.json"

    with open(json_path, "w", encoding="utf-8") as file:

        json.dump(
            landmarks_data,
            file,
            indent=4
        )

    print(f"\nSaved JSON -> {json_path}")

    # ------------------------------------------------------
    # 10. Draw pose landmarks
    # ------------------------------------------------------
    annotated_image = original_image.copy()

    connections = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),

        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),

        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),

        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),

        ("LEFT_HIP", "RIGHT_HIP"),

        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),

        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ]

    # ------------------------------------------------------
    # 11. Draw connections
    # ------------------------------------------------------
    for start_name, end_name in connections:

        start = landmarks_data[start_name]
        end = landmarks_data[end_name]

        start_point = (
            start["x_pixel"],
            start["y_pixel"]
        )

        end_point = (
            end["x_pixel"],
            end["y_pixel"]
        )

        cv2.line(
            annotated_image,
            start_point,
            end_point,
            (0, 255, 0),
            2
        )

    # ------------------------------------------------------
    # 12. Draw all landmarks
    # ------------------------------------------------------
    for name, data in landmarks_data.items():

        point = (
            data["x_pixel"],
            data["y_pixel"]
        )

        cv2.circle(
            annotated_image,
            point,
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            annotated_image,
            name,
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # ------------------------------------------------------
    # 13. Save visualized image
    # ------------------------------------------------------
    output_image_path = f"{output_dir}/pose_visualization.png"

    cv2.imwrite(
        output_image_path,
        annotated_image
    )

    print(f"Saved visualization -> {output_image_path}")

    return landmarks_data


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
if __name__ == "__main__":

    run_media_pose(
        PERSON_IMAGE,
        POSE_OUTPUT_DIR
    )