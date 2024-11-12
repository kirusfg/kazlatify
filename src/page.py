import os
import cv2
import numpy as np
import pytesseract
from typing import List, Tuple
from dataclasses import dataclass
from .test import ImageToWordModel
from .transformers import ImageThresholding
from mltu.configs import BaseModelConfigs


@dataclass
class TextBlock:
    image: np.ndarray
    box: Tuple[int, int, int, int]  # x, y, w, h
    line_num: int
    word_num: int


def extract_text_blocks(image: np.ndarray) -> List[TextBlock]:
    """Extract text blocks from image using Tesseract with custom configuration"""

    # Configuration string for Tesseract
    custom_config = r"--oem 3 --psm 3"  # Base configuration for page segmentation

    # Additional parameters to improve segmentation
    # custom_config += r" -c tessedit_char_blacklist=§±£∞"  # Remove problematic characters
    # custom_config += r" -c textord_max_noise_size=0.5"  # Reduce noise detection
    # custom_config += r" -c textord_noise_sizelimit=0.5"
    # custom_config += r" -c textord_min_linesize=30"  # Minimum text line size
    # custom_config += r" -c preserve_interword_spaces=1"  # Preserve space between words
    # custom_config += r" -c textord_min_xheight=30"  # Minimum x-height for text
    custom_config += r" -c textord_words_maxspace=120"  # Maximum space between words
    custom_config += r" -c textord_words_minspace=30"  # Minimum space between words
    custom_config += r" -c textord_words_default_maxspace=120"  # Default maximum word spacing
    custom_config += r" -c textord_words_default_minspace=30"  # Default minimum word spacing
    custom_config += r" -c textord_force_make_prop_words=F"  # Don't force proportional words

    # Pre-process image to improve segmentation
    # Increase contrast
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Optional: Additional image preprocessing
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Get bounding boxes from Tesseract with custom configuration
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)

    blocks = []
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        # Skip empty boxes or boxes with very low confidence
        if int(data["conf"][i]) < 0:  # Increased confidence threshold
            continue

        # Get coordinates
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]

        # Skip boxes that are too small or too large
        if w < 10 or h < 10 or w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
            continue

        # Calculate aspect ratio and skip extreme values
        aspect_ratio = w / h
        if aspect_ratio > 20 or aspect_ratio < 0.05:  # Adjust these thresholds as needed
            continue

        # Extract the region
        region = image[y : y + h, x : x + w]

        block = TextBlock(image=region, box=(x, y, w, h), line_num=data["line_num"][i], word_num=data["word_num"][i])
        blocks.append(block)

    # Additional filtering: merge very close blocks
    blocks = merge_close_blocks(blocks)

    # Sort blocks by line number and word number
    blocks.sort(key=lambda x: (x.line_num, x.word_num))
    return blocks


def merge_close_blocks(blocks: List[TextBlock], distance_threshold: int = 30, debug: bool = False) -> List[TextBlock]:
    """Merge blocks that are very close to each other"""
    if debug:
        print(f"Distance threshold: {distance_threshold}")

    if not blocks:
        return blocks

    merged_blocks = []
    current_block = blocks[0]

    for next_block in blocks[1:]:
        if debug:
            print(f"Comparing {current_block.line_num}_{current_block.word_num} and {next_block.line_num}_{next_block.word_num}")
            print(f"Distance: {next_block.box[0] - (current_block.box[0] + current_block.box[2])}")
            print(f"On same line: {current_block.line_num == next_block.line_num}")

        # If blocks are on the same line and close enough
        if (
            current_block.line_num == next_block.line_num
            and next_block.box[0] - (current_block.box[0] + current_block.box[2]) < distance_threshold
        ):
            # Merge blocks
            new_x = min(current_block.box[0], next_block.box[0])
            new_y = min(current_block.box[1], next_block.box[1])
            new_w = max(current_block.box[0] + current_block.box[2], next_block.box[0] + next_block.box[2]) - new_x
            new_h = max(current_block.box[1] + current_block.box[3], next_block.box[1] + next_block.box[3]) - new_y

            # Create merged block
            current_block = TextBlock(
                image=None,  # Will be updated in process_image
                box=(new_x, new_y, new_w, new_h),
                line_num=current_block.line_num,
                word_num=current_block.word_num,
            )

            if debug:
                print(f"Merged blocks: {current_block.line_num}_{current_block.word_num} and {next_block.line_num}_{next_block.word_num}")
        else:
            merged_blocks.append(current_block)
            current_block = next_block

    merged_blocks.append(current_block)
    return merged_blocks


def try_different_psm_modes(image: np.ndarray) -> List[TextBlock]:
    """Try different PSM modes and return the best result"""
    psm_modes = [
        3,  # Try automatic first
        4,  # Then single column
        6,  # Then single block
        7,  # Then single line
        11,  # Then sparse text
        1,  # Finally, try with OSD
    ]

    best_blocks = []
    max_blocks = 0
    best_psm_mode = 0

    for psm_mode in psm_modes:
        custom_config = f"--oem 3 --psm {psm_mode} -l kaz"
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)

        blocks = []
        for i in range(len(data["text"])):
            if int(data["conf"][i]) < 0:
                continue

            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]

            if w < 10 or h < 10 or w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
                continue

            blocks.append(
                TextBlock(image=image[y : y + h, x : x + w], box=(x, y, w, h), line_num=data["line_num"][i], word_num=data["word_num"][i])
            )

        if len(blocks) > max_blocks:
            max_blocks = len(blocks)
            best_blocks = blocks
            best_psm_mode = psm_mode

    print(f"Best PSM mode: {best_psm_mode}")

    # Additional filtering: merge very close blocks
    best_blocks = merge_close_blocks(best_blocks)

    # Sort blocks by line number and word number
    best_blocks.sort(key=lambda x: (x.line_num, x.word_num))

    return best_blocks


def process_image(image_path: str, model: ImageToWordModel, min_block_size: int = 20, padding: int = 0, save_segments: bool = False) -> str:
    """
    Process an image through text block detection and OCR

    Args:
        image_path: Path to input image
        model: Initialized ImageToWordModel
        min_block_size: Minimum size for text blocks
        padding: Padding to add around text blocks
        save_segments: Whether to save detected segments as images

    Returns:
        Reconstructed text from the image
    """
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Extract text blocks
    blocks = try_different_psm_modes(binary)

    if len(blocks) < 2:
        print("2")
        blocks = extract_text_blocks(binary)

    # If we still have too few blocks, try with different preprocessing
    if len(blocks) < 2:
        print("3")
        # Try with different preprocessing
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        blocks = extract_text_blocks(eroded)

    # Create directory for segments if saving is enabled
    if save_segments:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        segments_dir = f"segments_{base_name}"
        os.makedirs(segments_dir, exist_ok=True)

        # Save the original image
        cv2.imwrite(os.path.join(segments_dir, "original.png"), image)

        # Save intermediate images for debugging
        cv2.imwrite(os.path.join(segments_dir, "binary.png"), binary)
        cv2.imwrite(os.path.join(segments_dir, "gray.png"), gray)

        # Create visualization of all blocks
        viz_image = image.copy()
        for idx, block in enumerate(blocks):
            x, y, w, h = block.box
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(viz_image, f"{block.line_num}_{block.word_num}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(segments_dir, "blocks_visualization.png"), viz_image)

    # Process each block
    lines = []
    current_line = []
    current_line_num = -1

    for idx, block in enumerate(blocks):
        # Add padding to block
        x, y, w, h = block.box
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        # Extract padded region
        # print(image.shape)
        # print(gray.shape)
        region = image[y_start:y_end, x_start:x_end]
        padded_region = cv2.copyMakeBorder(region, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # print(region.shape)

        # Skip if region is too small
        if padded_region.shape[0] < min_block_size or padded_region.shape[1] < min_block_size:
            continue

        # Save segment if enabled
        if save_segments:
            segment_filename = f"segment_{block.line_num:02d}_{block.word_num:02d}.png"
            cv2.imwrite(os.path.join(segments_dir, segment_filename), padded_region)

        padded_region, _ = ImageThresholding()(padded_region, None)

        # Get predictions
        pred_text, pred_text_wbs = model.predict(padded_region)

        # Use WBS prediction if available, otherwise use regular prediction
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

    parser = argparse.ArgumentParser(description="Process images using text block detection and OCR")
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

        try:
            result = process_image(image_path, model, save_segments=args.save_segments)
            print(f"\nResults for {image_path}:")
            print("-" * 40)
            print(result)
            print("-" * 40)

            if args.save_segments:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                print(f"Segments saved in: segments_{base_name}/")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


if __name__ == "__main__":
    main()
