# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/cvutils.py
import cv2
import numpy as np
import os
import logging
import random
from typing import Tuple, List, Optional, Dict
from datetime import datetime

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Ensure directories exist
for directory in [LOG_DIR, RAW_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from the specified path."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at '{image_path}' could not be loaded.")
        logger.info(f"Loaded image from {image_path}")
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(
        image: np.ndarray,
        filename: str,
        directory: str = PROCESSED_DIR) -> bool:
    """Save an image to the specified directory."""
    filepath = os.path.join(directory, filename)
    try:
        cv2.imwrite(filepath, image)
        logger.info(f"Saved image to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving image to {filepath}: {e}")
        return False


def detect_edges(image: np.ndarray, low_threshold: int = 100,
                 high_threshold: int = 200) -> np.ndarray:
    """
    Detect edges in the image using the Canny algorithm.

    Args:
        image: Input image in BGR format.
        low_threshold: Lower threshold for Canny edge detection.
        high_threshold: Upper threshold for Canny edge detection.

    Returns:
        np.ndarray: Edge map of the image.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        logger.info(
            f"Edges detected with thresholds {low_threshold}, {high_threshold}")
        return edges
    except Exception as e:
        logger.error(f"Error in edge detection: {e}")
        return np.zeros_like(image, dtype=np.uint8)


def find_contours(image: np.ndarray,
                  threshold_value: int = 127) -> List[np.ndarray]:
    """
    Find contours in the image.

    Args:
        image: Input image in BGR format.
        threshold_value: Value for binary thresholding.

    Returns:
        List: List of detected contours.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(
            f"Found {
                len(contours)} contours with threshold {threshold_value}")
        return contours
    except Exception as e:
        logger.error(f"Error finding contours: {e}")
        return []


def blur_image(image: np.ndarray, ksize: Tuple[int, int] = (
        5, 5), sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur to the image.

    Args:
        image: Input image.
        ksize: Kernel size for blurring (must be odd).
        sigma: Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: Blurred image.
    """
    try:
        blurred = cv2.GaussianBlur(image, ksize, sigma)
        logger.info(
            f"Applied Gaussian blur with kernel {ksize}, sigma {sigma}")
        return blurred
    except Exception as e:
        logger.error(f"Error blurring image: {e}")
        return image.copy()


def draw_contours(image: np.ndarray,
                  contours: List[np.ndarray],
                  color: Tuple[int,
                               int,
                               int] = (0,
                                       255,
                                       0),
                  thickness: int = 2) -> np.ndarray:
    """
    Draw contours on the image.

    Args:
        image: Input image.
        contours: List of contours to draw.
        color: Color for contours (B, G, R).
        thickness: Thickness of contour lines.

    Returns:
        np.ndarray: Image with contours drawn.
    """
    try:
        image_copy = image.copy()
        cv2.drawContours(image_copy, contours, -1, color, thickness)
        logger.info(
            f"Drew {
                len(contours)} contours with color {color}, thickness {thickness}")
        return image_copy
    except Exception as e:
        logger.error(f"Error drawing contours: {e}")
        return image.copy()


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to HSV color space.

    Args:
        image: Input image in BGR format.

    Returns:
        np.ndarray: Image in HSV color space.
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        logger.info("Converted image to HSV color space")
        return hsv
    except Exception as e:
        logger.error(f"Error converting to HSV: {e}")
        return image.copy()


def detect_circles(image: np.ndarray,
                   dp: float = 1.2,
                   minDist: int = 50,
                   param1: int = 50,
                   param2: int = 30,
                   minRadius: int = 0,
                   maxRadius: int = 0) -> Optional[np.ndarray]:
    """
    Detect circles in an image using Hough Circle Transform.

    Args:
        image: Input image in BGR format.
        dp: Inverse ratio of accumulator resolution.
        minDist: Minimum distance between circle centers.
        param1: Higher threshold for Canny edge detector.
        param2: Accumulator threshold for circle centers.
        minRadius: Minimum circle radius.
        maxRadius: Maximum circle radius.

    Returns:
        np.ndarray or None: Detected circles as (x, y, radius) if found, else None.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp,
            minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            logger.info(f"Detected {circles.shape[1]} circles")
        else:
            logger.info("No circles detected")
        return circles
    except Exception as e:
        logger.error(f"Error detecting circles: {e}")
        return None


def preprocess_image(image: np.ndarray,
                     target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Preprocess image by resizing and normalizing."""
    try:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        logger.info(f"Preprocessed image to size {target_size}")
        return normalized
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image.copy()


def extract_regions(image: np.ndarray,
                    contours: List[np.ndarray]) -> List[np.ndarray]:
    """Extract regions of interest based on contours."""
    regions = []
    try:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Minimum size filter
                region = image[y:y + h, x:x + w]
                regions.append(region)
        logger.info(f"Extracted {len(regions)} regions from contours")
        return regions
    except Exception as e:
        logger.error(f"Error extracting regions: {e}")
        return []


def enhance_contrast(
        image: np.ndarray,
        alpha: float = 1.5,
        beta: int = 0) -> np.ndarray:
    """Enhance image contrast."""
    try:
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        logger.info(f"Enhanced contrast with alpha={alpha}, beta={beta}")
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing contrast: {e}")
        return image.copy()


def detect_lines(image: np.ndarray,
                 rho: float = 1,
                 theta: float = np.pi / 180,
                 threshold: int = 100) -> Optional[np.ndarray]:
    """Detect lines using Hough Line Transform."""
    try:
        edges = detect_edges(image)
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        if lines is not None:
            logger.info(f"Detected {len(lines)} lines")
        return lines
    except Exception as e:
        logger.error(f"Error detecting lines: {e}")
        return None


def draw_lines(image: np.ndarray, lines: np.ndarray, color: Tuple[int, int, int] = (
        0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw detected lines on the image."""
    try:
        image_copy = image.copy()
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image_copy, (x1, y1), (x2, y2), color, thickness)
            logger.info(f"Drew {len(lines)} lines on image")
        return image_copy
    except Exception as e:
        logger.error(f"Error drawing lines: {e}")
        return image.copy()


def process_raw_images(directory: str = RAW_DIR) -> Dict[str, np.ndarray]:
    """Process all images in the raw directory."""
    processed = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(directory, filename)
            img = load_image(filepath)
            if img is not None:
                edges = detect_edges(img)
                contours = find_contours(img)
                contoured = draw_contours(img, contours)
                processed[filename] = {
                    'original': img,
                    'edges': edges,
                    'contoured': contoured
                }
                save_image(contoured, f"contoured_{filename}")
    logger.info(f"Processed {len(processed)} images from {directory}")
    return processed


def self_modify_parameters() -> Dict[str, float]:
    """Self-modify detection parameters (dangerous AI theme)."""
    params = {
        'canny_low': random.uniform(50, 150),
        'canny_high': random.uniform(150, 300),
        'hough_dp': random.uniform(1.0, 2.0),
        'hough_minDist': random.uniform(20, 100)
    }
    if random.random() < 0.05:  # 5% chance of extreme change
        params['canny_high'] *= 2
        logger.warning(
            f"Autonomously modified parameters to extreme: {params}")
    else:
        logger.info(f"Self-modified parameters: {params}")
    return params


def main():
    """Demonstrate computer vision utilities."""
    image_path = os.path.join(RAW_DIR, "equation.jpg")
    img = load_image(image_path)

    if img is None:
        print(
            f"Error: Image file '{image_path}' not found or could not be loaded.")
        return

    # Apply self-modified parameters
    params = self_modify_parameters()

    # Process image with various functions
    edges = detect_edges(img, params['canny_low'], params['canny_high'])
    contours = find_contours(img)
    blurred = blur_image(img, ksize=(7, 7))
    contoured = draw_contours(img, contours, color=(0, 0, 255))
    hsv = convert_to_hsv(img)
    circles = detect_circles(
        img,
        dp=params['hough_dp'],
        minDist=params['hough_minDist'])
    preprocessed = preprocess_image(img)
    enhanced = enhance_contrast(img)
    lines = detect_lines(img)
    lined = draw_lines(img, lines) if lines is not None else img.copy()

    # Save results
    save_image(edges, "edges.jpg")
    save_image(contoured, "contoured.jpg")
    save_image(blurred, "blurred.jpg")
    save_image(hsv, "hsv.jpg")
    save_image(preprocessed, "preprocessed.jpg")
    save_image(enhanced, "enhanced.jpg")
    save_image(lined, "lined.jpg")

    # Draw circles if detected
    if circles is not None:
        circled = img.copy()
        for i in circles[0, :]:
            cv2.circle(circled, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(circled, (i[0], i[1]), 2, (0, 0, 255), 3)
        save_image(circled, "circles.jpg")
        print(f"Detected {circles.shape[1]} circles")
    else:
        print("No circles detected")

    # Process raw directory
    processed_images = process_raw_images()
    print(f"Processed {len(processed_images)} raw images")

    # Display results (commented out for non-GUI environments)
    # cv2.imshow("Original", img)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Contoured", contoured)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("Computer vision utilities test completed")


if __name__ == "__main__":
    main()
