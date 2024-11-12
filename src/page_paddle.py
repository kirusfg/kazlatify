import os
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from paddleocr import PaddleOCR
from .test import ImageToWordModel
from .transformers import ImageThresholding
from mltu.configs import BaseModelConfigs


@dataclass
class TextBlock:
    image: np.ndarray
    box: Tuple[int, int, int, int]  # x, y, w, h
    line_num: int
    word_num: int


def extract_text_blocks(image: np.ndarray, save_debug: bool = False) -> List[TextBlock]:
    """Extract text blocks using PaddleOCR's detection model"""

    # Initialize PaddleOCR with detection only
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="ru",
        rec=False,
        show_log=False,
        det=True,
        det_algorithm="DB",  # Detection algorithm
        det_limit_side_len=960,  # Max size for detection
        det_limit_type="max",
        det_db_thresh=0.1,  # Lower threshold to detect more text (default is 0.3)
        det_db_box_thresh=0.2,  # Lower threshold for box confidence (default is 0.5)
        det_db_unclip_ratio=1,  # Larger value to merge closer boxes (default is 1.6)
    )

    # Print image shape for debugging
    print(f"Image shape: {image.shape}")

    # Get detection results
    result = ocr.ocr(image, cls=False)
    print(f"Number of detected boxes: {len(result[0]) if result and result[0] else 0}")

    # Add debug visualization of raw detections
    if save_debug:
        debug_raw = image.copy()
        if len(debug_raw.shape) == 2:
            debug_raw = cv2.cvtColor(debug_raw, cv2.COLOR_GRAY2BGR)

        if result and result[0]:
            for item in result[0]:
                points = np.array(item[0]).astype(np.int32)
                cv2.polylines(debug_raw, [points], True, (0, 0, 255), 2)
        cv2.imwrite("paddle_raw_detection.png", debug_raw)

    if not result or not result[0]:
        return []

    # Extract just the boxes from the complex structure
    boxes = [item[0] for item in result[0]]  # Get just the coordinates, ignore the (text, confidence) tuples

    # Convert boxes to TextBlocks
    blocks = []
    current_line = 0
    current_word = 0
    last_y = -1
    line_height_threshold = 100  # Adjust based on your images

    # Sort boxes by y-coordinate first, then x-coordinate
    # boxes.sort(
    #     key=lambda box: (
    #         sum(point[1] for point in box) / 4,  # Average y-coordinate
    #         sum(point[0] for point in box) / 4,  # Average x-coordinate
    #     )
    # )

    for box in boxes:
        # Calculate center y-coordinate
        center_y = sum(point[1] for point in box) / 4

        # Check if this is a new line
        if last_y == -1 or abs(center_y - last_y) > line_height_threshold:
            current_line += 1
            current_word = 0
            last_y = center_y

        current_word += 1

        # Convert points to x,y,w,h format
        points = np.array(box).astype(np.int32)
        x = min(points[:, 0])
        y = min(points[:, 1])
        w = max(points[:, 0]) - x
        h = max(points[:, 1]) - y

        # Extract region
        region = image[y : y + h, x : x + w]

        blocks.append(TextBlock(image=region, box=(x, y, w, h), line_num=current_line, word_num=current_word))

    if save_debug:
        debug_img = image.copy()
        if len(debug_img.shape) == 2:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

        for block in blocks:
            x, y, w, h = block.box
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{block.line_num}_{block.word_num}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite("paddle_detection_debug.png", debug_img)

    return blocks


def process_image(image_path: str, model: ImageToWordModel, min_block_size: int = 20, padding: int = 5, save_segments: bool = False) -> str:
    """Process an image using PaddleOCR for detection and custom model for recognition"""

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Create directory for segments if saving is enabled
    if save_segments:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        segments_dir = f"segments_paddle_{base_name}"
        os.makedirs(segments_dir, exist_ok=True)

        # Save original image
        cv2.imwrite(os.path.join(segments_dir, "original.png"), image)

    # Extract text blocks
    blocks = extract_text_blocks(image, save_debug=save_segments)

    if not blocks:
        return ""

    # Process each block
    lines = []
    current_line = []
    current_line_num = -1

    for idx, block in enumerate(blocks):
        # Get region with padding
        x, y, w, h = block.box
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        region = image[y_start:y_end, x_start:x_end]

        # Add extra padding for the model
        padded_region = cv2.copyMakeBorder(region, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Skip if region is too small
        if padded_region.shape[0] < min_block_size or padded_region.shape[1] < min_block_size:
            continue

        # Save segment if enabled
        if save_segments:
            segment_filename = f"segment_{block.line_num:02d}_{block.word_num:02d}.png"
            cv2.imwrite(os.path.join(segments_dir, segment_filename), padded_region)

        # Preprocess for model
        padded_region, _ = ImageThresholding()(padded_region, None)

        # Get predictions
        pred_text, pred_text_wbs = model.predict(padded_region)

        # Use WBS prediction if available
        text = pred_text_wbs if pred_text_wbs else pred_text

        # Save prediction if enabled
        if save_segments:
            with open(os.path.join(segments_dir, "predictions.txt"), "a", encoding="utf-8") as f:
                f.write(f"Segment {block.line_num:02d}_{block.word_num:02d}:\n")
                f.write(f"Regular prediction: {pred_text}\n")
                f.write(f"WBS prediction: {pred_text_wbs}\n")
                f.write("-" * 40 + "\n")

        # Handle line breaks
        if block.line_num != current_line_num:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = []
            current_line_num = block.line_num

        current_line.append(text)

    # Add the last line
    if current_line:
        lines.append(" ".join(current_line))

    final_text = "\n".join(lines)

    # Save final text if enabled
    if save_segments:
        with open(os.path.join(segments_dir, "final_text.txt"), "w", encoding="utf-8") as f:
            f.write(final_text)

    return final_text


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process images using PaddleOCR detection and custom recognition")
    parser.add_argument("--model", type=str, required=True, help="Model to use (HKR, KOHTD, HKR+KOHTD)")
    parser.add_argument("--images", type=str, required=True, help="Comma-separated list of image paths")
    parser.add_argument("--use_wbs", action="store_true", help="Use Word Beam Search for decoding")
    parser.add_argument("--save_segments", action="store_true", help="Save detected segments as individual images")

    args = parser.parse_args()

    # Model checkpoint mapping
    models_checkpoints = {
        "HKR": "hkr_202411051410",
        "KOHTD": "kohtd_202411051413",
        "HKR+KOHTD": "both_202411051413",
    }

    if args.model not in models_checkpoints:
        raise ValueError(f"Invalid model name. Choose from: {list(models_checkpoints.keys())}")

    # Load model
    model_checkpoint = models_checkpoints[args.model]
    configs = BaseModelConfigs.load(f"checkpoints/{model_checkpoint}/configs.yaml")

    # Set language based on model
    lang = "ru" if args.model == "HKR" else "kz" if args.model == "KOHTD" else "*"
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab, lang=lang, use_wbs=args.use_wbs)

    # Process each image
    images = [path.strip() for path in args.images.split(",")]
    for image_path in images:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
            continue

        result = process_image(image_path, model, save_segments=args.save_segments)
        print(f"\nResults for {image_path}:")
        print("-" * 40)
        print(result)
        print("-" * 40)

        if args.save_segments:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"Segments saved in: segments_paddle_{base_name}/")


if __name__ == "__main__":
    main()
