import cv2
import os
from pathlib import Path

def downsample_image(input_path, output_folder, scale_factor=0.5):
    """
    एक image वाचतो, तो scale_factor ने downsample करतो,
    आणि output_folder मध्ये save करतो.

    input_path    : ज्या image वर काम करायचं त्याचा पूर्ण path
    output_folder : downsampled image कुठे save करायची
    scale_factor  : किती लहान करायचं (0.5 म्हणजे अर्ध्या size ला)
    """





    # Step 1: image वाचा
    img = cv2.imread(input_path)

    if img is None:
        print(f"Error: image वाचता आली नाही -> {input_path}")
        return

    # Step 2: original size काढा
    original_height, original_width = img.shape[:2]

    # Step 3: नवीन size काढा
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Step 4: downsample करा (INTER_AREA = downsampling साठी सर्वोत्तम)
    downsampled_img = cv2.resize(
        img,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Step 5: output folder नसेल तर तयार करा
    os.makedirs(output_folder, exist_ok=True)

    # Step 6: file name काढा आणि save करा
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, f"downsampled_{filename}")
    cv2.imwrite(output_path, downsampled_img)

    print(f"Original size: {original_width}x{original_height}")
    print(f"Downsampled size: {new_width}x{new_height}")
    print(f"Saved at: {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]


    # ----- इथे तुमचे actual path टाका -----
    input_image_path = f"{BASE_DIR}/fabric_images/test_fabrics/design_green41.png"
    output_folder_path = f"{BASE_DIR}/test_images/downsample_1.png"
    scale = 0.3  # 50% size

    downsample_image(input_image_path, output_folder_path, scale)