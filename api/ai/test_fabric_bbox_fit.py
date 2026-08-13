import cv2
import numpy as np
import os
from pathlib import Path


def xyxy_to_xywh(bbox_xyxy):
    """
    GroundingDINO चा output (x1, y1, x2, y2) format मध्ये असतो
    (Hugging Face अधिकृत डॉक्युमेंटेशन नुसार: top_left_x, top_left_y,
    bottom_right_x, bottom_right_y).

    हे function त्याला (x, y, w, h) मध्ये convert करतं,
    आणि float coordinates ना pixel indexing साठी integer मध्ये round करतं.
    """

    x1, y1, x2, y2 = bbox_xyxy

    x = int(round(x1))
    y = int(round(y1))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))

    return x, y, w, h


def compute_uniform_scale(fabric_w, fabric_h, bbox_w, bbox_h):
    """
    Fabric ला bbox मध्ये पूर्णपणे (कुठेही रिकामी जागा न ठेवता) बसवण्यासाठी
    लागणारा एकच (uniform) scale काढतो. Aspect ratio कायम राहतो.
    """

    scale_x = bbox_w / fabric_w
    scale_y = bbox_h / fabric_h

    scale = max(scale_x, scale_y)

    return scale


def resize_fabric_uniform(fabric_img, scale):
    """
    Fabric ला दिलेल्या scale ने resize करतो (aspect ratio सांभाळून).
    """

    fabric_h, fabric_w = fabric_img.shape[:2]

    new_w = int(round(fabric_w * scale))
    new_h = int(round(fabric_h * scale))

    # scale लहान करताना (downsample) INTER_AREA,
    # मोठं करताना (upsample) INTER_LANCZOS4
    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LANCZOS4
    )

    resized = cv2.resize(
        fabric_img,
        (new_w, new_h),
        interpolation=interpolation
    )
    print('resized:', resized.shape)

    return resized


def center_crop_to_bbox(resized_fabric, bbox_w, bbox_h):
    """
    Resize केलेला fabric (जो bbox पेक्षा किंचित मोठा असू शकतो)
    मधोमध पकडून exact bbox size ला crop करतो.
    """

    resized_h, resized_w = resized_fabric.shape[:2]

    start_x = (resized_w - bbox_w) // 2
    start_y = (resized_h - bbox_h) // 2

    cropped = resized_fabric[
        start_y:start_y + bbox_h,
        start_x:start_x + bbox_w
    ]

    print("cropped shape: ", cropped.shape)

    return cropped


def create_black_canvas(person_h, person_w, channels):
    """
    Person image एवढ्याच size चा पूर्ण काळा canvas तयार करतो.
    """

    canvas = np.zeros(
        (person_h, person_w, channels),
        dtype=np.uint8
    )

    return canvas


def paste_fabric_on_canvas(canvas, cropped_fabric, bbox_x, bbox_y):
    """
    Cropped fabric ला black canvas वर bbox च्या (x, y)
    location वर paste करतो.
    """
    print('bbox x', bbox_x)
    print('bbox y', bbox_y)
    bbox_h, bbox_w = cropped_fabric.shape[:2]

    canvas[
        bbox_y:bbox_y + bbox_h,
        bbox_x:bbox_x + bbox_w
    ] = cropped_fabric

    return canvas


def draw_bbox_for_verification(person_img_path, bbox_xywh, output_folder):
    """
    Person image वर दिलेला bbox rectangle काढून save करतो,
    जेणेकरून bbox खऱ्या shirt location वर बरोबर बसतो का
    ते डोळ्यांनी तपासता येईल.
    """

    person_img = cv2.imread(person_img_path)

    if person_img is None:
        print(
            f"Error: person image वाचता आली नाही -> "
            f"{person_img_path}"
        )
        return

    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh

    img_with_box = person_img.copy()

    cv2.rectangle(
        img_with_box,
        (bbox_x, bbox_y),
        (bbox_x + bbox_w, bbox_y + bbox_h),
        (0, 0, 255),  # लाल रंग (BGR)
        thickness=3
    )

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "bbox_verification.png")

    cv2.imwrite(output_path, img_with_box)

    print(f"Verification image saved at: {output_path}")


def fit_fabric_to_bbox(
    person_img_path,
    fabric_img_path,
    bbox_xywh,
    output_folder
):
    """
    Main fabric-to-bounding-box fitting function.

    Fabric ला bbox च्या size मध्ये (aspect ratio सांभाळून) बसवून,
    person image एवढ्या size च्या काळ्या canvas वर bbox location वर
    paste करतो.
    """

    person_img = cv2.imread(person_img_path)
    fabric_img = cv2.imread(fabric_img_path)

    if person_img is None:
        print(f"ERROR: Person image वाचता आली नाही -> {person_img_path}")
        return

    if fabric_img is None:
        print(f"ERROR: Fabric image वाचता आली नाही -> {fabric_img_path}")
        return

    person_h, person_w, channels = person_img.shape
    fabric_h, fabric_w = fabric_img.shape[:2]
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh

    # Safety check: bbox person image च्या आत बसतो का
    if bbox_x < 0 or bbox_y < 0 or bbox_w <= 0 or bbox_h <= 0:
        print("ERROR: BBox coordinates invalid आहेत.")
        return

    if (bbox_x + bbox_w) > person_w or (bbox_y + bbox_h) > person_h:
        print("ERROR: BBox person image च्या बाहेर जातो आहे.")
        return

    # Step 1: uniform scale काढा
    scale = compute_uniform_scale(fabric_w, fabric_h, bbox_w, bbox_h)

    # Step 2: resize करा
    resized_fabric = resize_fabric_uniform(fabric_img, scale)

    # Step 3: bbox size ला exact crop करा
    cropped_fabric = center_crop_to_bbox(resized_fabric, bbox_w, bbox_h)

    # Step 4: black canvas तयार करा
    canvas = create_black_canvas(person_h, person_w, channels)

    # Step 5: cropped fabric bbox location वर paste करा
    final_output = paste_fabric_on_canvas(canvas, cropped_fabric, bbox_x, bbox_y)

    # Step 6: save करा
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "fabric_fitted_to_bbox4.png")
    cv2.imwrite(output_path, final_output)

    print(f"Saved at: {output_path}")


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[2]

    person_image_path = f"{BASE_DIR}/test_images/person16.jpeg"
    fabric_image_path = f"{BASE_DIR}/fabric_images/floral_fabric7.png"
    output_folder_path = f"{BASE_DIR}/test_images/bbox_fit_output"

    # तुमचा actual GroundingDINO output (x1, y1, x2, y2) format मध्ये
    groundingdino_bbox_xyxy = [
        362.3363037109375,
        672.457275390625,
        700.0819091796875,
        1113.4124755859375
    ]

    test_bbox_xywh = xyxy_to_xywh(groundingdino_bbox_xyxy)

    draw_bbox_for_verification(person_image_path, test_bbox_xywh, output_folder_path)

    fit_fabric_to_bbox(person_image_path, fabric_image_path, test_bbox_xywh, output_folder_path)