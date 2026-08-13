import cv2
import numpy as np
import os
from pathlib import Path


def xyxy_to_xywh(bbox_xyxy):
    """
    GroundingDINO चा output (x1, y1, x2, y2) format मध्ये असतो
    (Hugging Face अधिकृत डॉक्युमेंटेशन नुसार: top_left_x, top_left_y, bottom_right_x, bottom_right_y).
    हे function त्याला (x, y, w, h) मध्ये convert करतं, आणि float coordinates ना
    pixel indexing साठी integer मध्ये round करतं.
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

    # scale लहान करताना (downsample) INTER_AREA, मोठं करताना (upsample) INTER_LANCZOS4
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4

    resized = cv2.resize(fabric_img, (new_w, new_h), interpolation=interpolation)
    return resized


def center_crop_to_bbox(resized_fabric, bbox_w, bbox_h):
    """
    Resize केलेला fabric (जो bbox पेक्षा किंचित मोठा असू शकतो) मधोमध पकडून
    exact bbox size ला crop करतो.
    """
    resized_h, resized_w = resized_fabric.shape[:2]

    start_x = (resized_w - bbox_w) // 2
    start_y = (resized_h - bbox_h) // 2

    cropped = resized_fabric[start_y:start_y + bbox_h, start_x:start_x + bbox_w]
    return cropped


def create_black_canvas(person_h, person_w, channels):
    """
    Person image एवढ्याच size चा पूर्ण काळा canvas तयार करतो.
    """
    canvas = np.zeros((person_h, person_w, channels), dtype=np.uint8)
    return canvas


def paste_fabric_on_canvas(canvas, cropped_fabric, bbox_x, bbox_y):
    """
    Cropped fabric ला black canvas वर bbox च्या (x, y) location वर paste करतो.
    """
    bbox_h, bbox_w = cropped_fabric.shape[:2]
    canvas[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w] = cropped_fabric
    return canvas


def draw_bbox_for_verification(person_img_path, bbox_xywh, output_folder):
    """
    Person image वर दिलेला bbox rectangle काढून save करतो,
    जेणेकरून bbox खऱ्या shirt location वर बरोबर बसतो का ते डोळ्यांनी तपासता येईल.
    """
    BASE_DIR = Path(__file__).resolve().parents[2]
    person_img = cv2.imread(person_img_path)
    if person_img is None:
        print(f"Error: person image वाचता आली नाही -> {person_img_path}")
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
    output_path = os.path.join(output_folder, "bbox_verification.jpg")
    output_folder_path = f"{BASE_DIR}/test_images/bbox_fit_output/vrifacation_bbox.png"
    cv2.imwrite(output_folder_path, img_with_box)
    print(f"Verification image saved at: {output_path}")


def fit_fabric_to_bbox(person_img_path, fabric_img_path, bbox_xywh, output_folder):
    """
    Main fabric-to-bounding-box fitting function.

    DEBUG VERSION:
    प्रत्येक intermediate stage ची image save करते,
    जेणेकरून fabric कुठल्या stage ला चुकीची होत आहे
    हे visually verify करता येईल.
    """

    # ---------------------------------------------------------
    # STEP 1: Read images
    # ---------------------------------------------------------

    person_img = cv2.imread(person_img_path)
    fabric_img = cv2.imread(fabric_img_path)

    if person_img is None:
        print(f"ERROR: Person image वाचता आली नाही -> {person_img_path}")
        return

    if fabric_img is None:
        print(f"ERROR: Fabric image वाचता आली नाही -> {fabric_img_path}")
        return

    # ---------------------------------------------------------
    # STEP 2: Create debug folders
    # ---------------------------------------------------------

    os.makedirs(output_folder, exist_ok=True)

    debug_folder = os.path.join(
        output_folder,
        "debug_fit_fabric"
    )

    os.makedirs(debug_folder, exist_ok=True)

    print("\n" + "=" * 70)
    print("FIT FABRIC TO BBOX - DEBUG START")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 3: Get image dimensions
    # ---------------------------------------------------------

    person_h, person_w, channels = person_img.shape
    fabric_h, fabric_w = fabric_img.shape[:2]

    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh

    print("\n[PERSON IMAGE]")
    print(f"Width  : {person_w}")
    print(f"Height : {person_h}")
    print(f"Channels: {channels}")

    print("\n[FABRIC IMAGE - ORIGINAL]")
    print(f"Width  : {fabric_w}")
    print(f"Height : {fabric_h}")

    print("\n[BOUNDING BOX]")
    print(f"x      : {bbox_x}")
    print(f"y      : {bbox_y}")
    print(f"width  : {bbox_w}")
    print(f"height : {bbox_h}")

    print("\n[BBOX RIGHT/BOTTOM]")
    print(f"right  : {bbox_x + bbox_w}")
    print(f"bottom : {bbox_y + bbox_h}")

    # ---------------------------------------------------------
    # STEP 4: Safety check
    # ---------------------------------------------------------

    if bbox_x < 0 or bbox_y < 0:
        print("\nERROR: BBox चा x किंवा y negative आहे.")
        return

    if bbox_w <= 0 or bbox_h <= 0:
        print("\nERROR: BBox width किंवा height invalid आहे.")
        return

    if (bbox_x + bbox_w) > person_w:
        print("\nERROR: BBox right side person image च्या बाहेर आहे.")
        print(
            f"BBox right = {bbox_x + bbox_w}, "
            f"Person width = {person_w}"
        )
        return

    if (bbox_y + bbox_h) > person_h:
        print("\nERROR: BBox bottom person image च्या बाहेर आहे.")
        print(
            f"BBox bottom = {bbox_y + bbox_h}, "
            f"Person height = {person_h}"
        )
        return

    print("\nBBox पूर्णपणे person image च्या आत आहे.")

    # ---------------------------------------------------------
    # DEBUG IMAGE 1
    # Original person + bounding box
    # ---------------------------------------------------------

    person_with_bbox = person_img.copy()

    cv2.rectangle(
        person_with_bbox,
        (bbox_x, bbox_y),
        (bbox_x + bbox_w, bbox_y + bbox_h),
        (0, 0, 255),
        3
    )

    cv2.putText(
        person_with_bbox,
        f"BBOX: x={bbox_x}, y={bbox_y}, "
        f"w={bbox_w}, h={bbox_h}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    bbox_debug_path = os.path.join(
        debug_folder,
        "01_original_person_with_bbox.jpg"
    )

    cv2.imwrite(
        bbox_debug_path,
        person_with_bbox
    )

    print(f"\nDEBUG IMAGE 1 saved:")
    print(bbox_debug_path)

    # ---------------------------------------------------------
    # STEP 5: Calculate uniform scale
    # ---------------------------------------------------------

    scale = compute_uniform_scale(
        fabric_w,
        fabric_h,
        bbox_w,
        bbox_h
    )

    print("\n[UNIFORM SCALE]")
    print(f"Scale X = {bbox_w / fabric_w:.6f}")
    print(f"Scale Y = {bbox_h / fabric_h:.6f}")
    print(f"Selected Scale = {scale:.6f}")

    # ---------------------------------------------------------
    # STEP 6: Resize fabric
    # ---------------------------------------------------------

    resized_fabric = resize_fabric_uniform(
        fabric_img,
        scale
    )

    resized_h, resized_w = resized_fabric.shape[:2]

    print("\n[RESIZED FABRIC]")
    print(f"Width  : {resized_w}")
    print(f"Height : {resized_h}")

    # ---------------------------------------------------------
    # DEBUG IMAGE 2
    # Resized fabric
    # ---------------------------------------------------------

    resized_debug_path = os.path.join(
        debug_folder,
        "02_resized_fabric.jpg"
    )

    cv2.imwrite(
        resized_debug_path,
        resized_fabric
    )

    print("DEBUG IMAGE 2 saved:")
    print(resized_debug_path)

    # ---------------------------------------------------------
    # STEP 7: Center crop
    # ---------------------------------------------------------

    cropped_fabric = center_crop_to_bbox(
        resized_fabric,
        bbox_w,
        bbox_h
    )

    cropped_h, cropped_w = cropped_fabric.shape[:2]

    print("\n[CROPPED FABRIC]")
    print(f"Expected width  : {bbox_w}")
    print(f"Actual width    : {cropped_w}")
    print(f"Expected height : {bbox_h}")
    print(f"Actual height   : {cropped_h}")

    # ---------------------------------------------------------
    # VERY IMPORTANT DEBUG CHECK
    # ---------------------------------------------------------

    if cropped_w != bbox_w:
        print("\n!!! WIDTH MISMATCH !!!")
        print(
            f"Expected cropped width = {bbox_w}, "
            f"but got = {cropped_w}"
        )

    if cropped_h != bbox_h:
        print("\n!!! HEIGHT MISMATCH !!!")
        print(
            f"Expected cropped height = {bbox_h}, "
            f"but got = {cropped_h}"
        )

    # ---------------------------------------------------------
    # DEBUG IMAGE 3
    # Cropped fabric
    # ---------------------------------------------------------

    cropped_debug_path = os.path.join(
        debug_folder,
        "03_cropped_fabric_exact_bbox_size.jpg"
    )

    cv2.imwrite(
        cropped_debug_path,
        cropped_fabric
    )

    print("DEBUG IMAGE 3 saved:")
    print(cropped_debug_path)

    # ---------------------------------------------------------
    # STEP 8: Create black canvas
    # ---------------------------------------------------------

    canvas = create_black_canvas(
        person_h,
        person_w,
        channels
    )

    print("\n[BLACK CANVAS]")
    print(f"Width  : {canvas.shape[1]}")
    print(f"Height : {canvas.shape[0]}")

    # ---------------------------------------------------------
    # STEP 9: Paste fabric
    # ---------------------------------------------------------

    final_output = paste_fabric_on_canvas(
        canvas,
        cropped_fabric,
        bbox_x,
        bbox_y
    )

    final_h, final_w = final_output.shape[:2]

    print("\n[FINAL CANVAS]")
    print(f"Width  : {final_w}")
    print(f"Height : {final_h}")

    # ---------------------------------------------------------
    # DEBUG IMAGE 4
    # Fabric on black canvas
    # ---------------------------------------------------------

    canvas_debug_path = os.path.join(
        debug_folder,
        "04_fabric_on_black_canvas.jpg"
    )

    cv2.imwrite(
        canvas_debug_path,
        final_output
    )

    print("DEBUG IMAGE 4 saved:")
    print(canvas_debug_path)

    # ---------------------------------------------------------
    # DEBUG IMAGE 5
    # Overlay fabric canvas on ORIGINAL PERSON
    # ---------------------------------------------------------

    overlay = person_img.copy()

    # Fabric canvas मधील non-black pixels शोधा
    gray_canvas = cv2.cvtColor(
        final_output,
        cv2.COLOR_BGR2GRAY
    )

    fabric_region_mask = gray_canvas > 5

    # Fabric region person image वर दाखवा
    overlay[fabric_region_mask] = final_output[fabric_region_mask]

    # Bounding box पुन्हा draw करा
    cv2.rectangle(
        overlay,
        (bbox_x, bbox_y),
        (bbox_x + bbox_w, bbox_y + bbox_h),
        (0, 0, 255),
        3
    )

    cv2.putText(
        overlay,
        "RED = BBOX | FABRIC = ACTUAL PASTED REGION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2
    )

    overlay_debug_path = os.path.join(
        debug_folder,
        "05_PERSON_WITH_FABRIC_AND_BBOX.jpg"
    )

    cv2.imwrite(
        overlay_debug_path,
        overlay
    )

    print("DEBUG IMAGE 5 saved:")
    print(overlay_debug_path)

    # ---------------------------------------------------------
    # STEP 10: Save final output
    # ---------------------------------------------------------

    output_path = os.path.join(
        output_folder,
        "fabric_fitted_to_bbox.jpg"
    )

    cv2.imwrite(
        output_path,
        final_output
    )

    print("\nFINAL OUTPUT saved:")
    print(output_path)

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DEBUG SUMMARY")
    print("=" * 70)

    print(f"Person              : {person_w} x {person_h}")
    print(f"Original Fabric     : {fabric_w} x {fabric_h}")
    print(f"BBox                : {bbox_w} x {bbox_h}")
    print(f"Scale               : {scale:.6f}")
    print(f"Resized Fabric      : {resized_w} x {resized_h}")
    print(f"Cropped Fabric      : {cropped_w} x {cropped_h}")
    print(f"Final Canvas        : {final_w} x {final_h}")

    print("\nDebug images folder:")
    print(debug_folder)

    print("=" * 70)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    person_image_path = f"{BASE_DIR}/test_images/person18.jpeg"
    fabric_image_path = f"{BASE_DIR}/fabric_images/test_fabrics/design_green41.png"
    output_folder_path = f"{BASE_DIR}/test_images/bbox_fit_output"

    # ----- तुमचा actual GroundingDINO output (x1, y1, x2, y2) format मध्ये -----
    groundingdino_bbox_xyxy = [362.3363037109375, 672.457275390625, 700.0819091796875, 1113.4124755859375]

    # xyxy -> xywh convert करा (एकाच ठिकाणी, पूर्ण कोडमध्ये xywh च वापरला जातो)
    test_bbox_xywh = xyxy_to_xywh(groundingdino_bbox_xyxy)

    # Step A: आधी bbox बरोबर shirt वर बसतो का ते verify करा
    draw_bbox_for_verification(person_image_path, test_bbox_xywh, output_folder_path)

    # Step B: fabric ला त्या bbox मध्ये fit करा
    fit_fabric_to_bbox(person_image_path, fabric_image_path, test_bbox_xywh, output_folder_path)