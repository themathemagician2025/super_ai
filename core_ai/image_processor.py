# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/image_processor.py
import os
import cv2
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union, Dict, List
from datetime import datetime
import pickle
from config import PROJECT_CONFIG, get_project_config
from data_loader import load_raw_data, save_processed_data
from helpers import normalize_data, scale_data

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
IMG_DIR = os.path.join(DATA_DIR, 'images')  # New directory for images

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, MODELS_DIR, IMG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load project configuration
CONFIG = get_project_config()


class ImageProcessor:
    """Advanced image processing class with self-modification."""

    def __init__(self):
        self.modification_count = 0
        self.max_modifications = CONFIG["self_modification"]["max_mutations"]
        self.processing_history = []
        self.default_size = (256, 256)
        self.blur_kernel = (5, 5)
        self.canny_thresholds = (100, 200)

    def load_image(self, path: str) -> np.ndarray:
        """
        Load an image from the given path.

        Args:
            path: File path to the image.

        Returns:
            Loaded image array.

        Raises:
            FileNotFoundError, ValueError: If loading fails.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Failed to load image from: {path}")
            logger.info("Loaded image from %s", path)
            return image
        except Exception as e:
            logger.error("Error loading image: %s", e)
            raise

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self._log_processing("grayscale", image.shape)
            return gray
        except Exception as e:
            logger.error("Error converting to grayscale: %s", e)
            raise

    def resize_image(self, image: np.ndarray,
                     size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize image to a specified size.

        Args:
            image: Input image.
            size: Desired size (width, height); uses default if None.

        Returns:
            Resized image.
        """
        try:
            size = size or self.default_size
            resized = cv2.resize(image, size)
            self._log_processing("resize", image.shape, {"new_size": size})
            return resized
        except Exception as e:
            logger.error("Error resizing image: %s", e)
            raise

    def detect_edges(
            self,
            image: np.ndarray,
            threshold1: int = None,
            threshold2: int = None) -> np.ndarray:
        """
        Detect edges using Canny algorithm with self-modifying thresholds.

        Args:
            image: Input image.
            threshold1: First threshold (optional).
            threshold2: Second threshold (optional).

        Returns:
            Edge map.
        """
        try:
            t1, t2 = threshold1 or self.canny_thresholds[0], threshold2 or self.canny_thresholds[1]
            if CONFIG["self_modification"]["enabled"]:
                t1, t2 = self._self_modify_canny(t1, t2)
            gray = self.grayscale(image)
            edges = cv2.Canny(gray, t1, t2)
            self._log_processing(
                "detect_edges", image.shape, {
                    "t1": t1, "t2": t2})
            return edges
        except Exception as e:
            logger.error("Error detecting edges: %s", e)
            raise

    def blur_image(self,
                   image: np.ndarray,
                   ksize: Tuple[int,
                                int] = None) -> np.ndarray:
        """
        Apply Gaussian blur with self-modifying kernel.

        Args:
            image: Input image.
            ksize: Kernel size (optional).

        Returns:
            Blurred image.
        """
        try:
            ksize = ksize or self.blur_kernel
            if CONFIG["self_modification"]["enabled"]:
                ksize = self._self_modify_kernel(ksize)
            blurred = cv2.GaussianBlur(image, ksize, 0)
            self._log_processing("blur", image.shape, {"ksize": ksize})
            return blurred
        except Exception as e:
            logger.error("Error blurring image: %s", e)
            raise

    def _self_modify_canny(self, t1: int, t2: int) -> Tuple[int, int]:
        """Self-modify Canny thresholds (dangerous AI theme)."""
        if (random.random() < CONFIG["self_modification"]["autonomous_rate"]
                and self.modification_count < self.max_modifications):
            self.modification_count += 1
            t1 = int(t1 * random.uniform(0.8, 1.2))
            t2 = int(t2 * random.uniform(0.8, 1.2))
            t1, t2 = min(max(t1, 50), 300), min(max(t2, 100), 400)  # Constrain
            self.canny_thresholds = (t1, t2)
            logger.warning(
                "Self-modified Canny thresholds to (%d, %d)", t1, t2)
        return t1, t2

    def _self_modify_kernel(self, ksize: Tuple[int, int]) -> Tuple[int, int]:
        """Self-modify blur kernel size."""
        if (random.random() < CONFIG["self_modification"]["autonomous_rate"]
                and self.modification_count < self.max_modifications):
            self.modification_count += 1
            k = random.choice([3, 5, 7, 9])
            ksize = (k, k)  # Ensure odd and square
            self.blur_kernel = ksize
            logger.warning("Self-modified blur kernel to %s", ksize)
        return ksize

    def _log_processing(
            self,
            operation: str,
            shape: Tuple,
            params: Dict = None) -> None:
        """Log processing step."""
        self.processing_history.append({
            "operation": operation,
            "timestamp": datetime.now(),
            "shape": shape,
            "params": params or {}
        })
        logger.info(
            "Processed image with %s: shape %s, params %s",
            operation,
            shape,
            params)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values using helpers.py."""
        try:
            norm_image, _, _ = normalize_data(image.astype(float))
            return norm_image.astype(np.uint8)
        except Exception as e:
            logger.error("Error normalizing image: %s", e)
            raise

    def enhance_contrast(
            self,
            image: np.ndarray,
            alpha: float = 1.5,
            beta: float = 0) -> np.ndarray:
        """Enhance image contrast."""
        try:
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            self._log_processing(
                "enhance_contrast", image.shape, {
                    "alpha": alpha, "beta": beta})
            return enhanced
        except Exception as e:
            logger.error("Error enhancing contrast: %s", e)
            raise

    def detect_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect contours in the image."""
        try:
            edges = self.detect_edges(image)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self._log_processing(
                "detect_contours", image.shape, {
                    "count": len(contours)})
            return contours
        except Exception as e:
            logger.error("Error detecting contours: %s", e)
            raise

    def save_image(self, image: np.ndarray, filename: str) -> bool:
        """Save processed image to disk."""
        filepath = os.path.join(IMG_DIR, filename)
        try:
            cv2.imwrite(filepath, image)
            logger.info("Image saved to %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving image to %s: %s", filepath, e)
            return False

    def process_batch(self,
                      image_paths: List[str],
                      operations: List[str]) -> Dict[str,
                                                     np.ndarray]:
        """Process a batch of images with specified operations."""
        results = {}
        for path in image_paths:
            try:
                img = self.load_image(path)
                for op in operations:
                    if op == "grayscale":
                        img = self.grayscale(img)
                    elif op == "resize":
                        img = self.resize_image(img)
                    elif op == "edges":
                        img = self.detect_edges(img)
                    elif op == "blur":
                        img = self.blur_image(img)
                    elif op == "contrast":
                        img = self.enhance_contrast(img)
                results[os.path.basename(path)] = img
                self.save_image(img, f"processed_{os.path.basename(path)}")
            except Exception as e:
                logger.error("Error processing %s: %s", path, e)
        return results


def load_image(path: str) -> np.ndarray:
    """Wrapper for standalone use."""
    return ImageProcessor().load_image(path)


def grayscale(image: np.ndarray) -> np.ndarray:
    """Wrapper for standalone use."""
    return ImageProcessor().grayscale(image)


def resize_image(image: np.ndarray,
                 size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Wrapper for standalone use."""
    return ImageProcessor().resize_image(image, size)


def detect_edges(
        image: np.ndarray,
        threshold1: int = 100,
        threshold2: int = 200) -> np.ndarray:
    """Wrapper for standalone use."""
    return ImageProcessor().detect_edges(image, threshold1, threshold2)


def blur_image(image: np.ndarray,
               ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Wrapper for standalone use."""
    return ImageProcessor().blur_image(image, ksize)


def display_image(window_name: str, image: np.ndarray) -> None:
    """Display an image in a window."""
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error("Error displaying image: %s", e)
        raise


def main():
    """Demonstrate image processing functionality."""
    processor = ImageProcessor()
    image_path = os.path.join(IMG_DIR, "equation.jpg")

    # Simulate an image if not present
    if not os.path.exists(image_path):
        dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy_img)
        logger.info("Created dummy image at %s", image_path)

    try:
        # Load and process
        img = processor.load_image(image_path)
        resized = processor.resize_image(img)
        gray = processor.grayscale(resized)
        edges = processor.detect_edges(resized)
        blurred = processor.blur_image(resized)
        contrast = processor.enhance_contrast(resized)
        contours = processor.detect_contours(resized)

        # Batch processing
        batch_paths = [image_path]
        batch_results = processor.process_batch(
            batch_paths, ["resize", "edges", "blur"])

        # Display results
        display_image("Original", resized)
        display_image("Grayscale", gray)
        display_image("Edges", edges)
        display_image("Blurred", blurred)
        display_image("Contrast", contrast)

        # Log history
        logger.info("Processing history: %s", processor.processing_history)

    except Exception as e:
        logger.error("Error in demo: %s", e)

    logger.info("Image processor demo completed.")


if __name__ == "__main__":
    main()

# Utilities


def save_processor_state(processor: ImageProcessor, filename: str) -> bool:
    """Save ImageProcessor state."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(processor.__dict__, f)
        logger.info("Processor state saved to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving processor state: %s", e)
        return False


def load_processor_state(filename: str) -> Optional[ImageProcessor]:
    """Load ImageProcessor state."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        processor = ImageProcessor()
        with open(filepath, 'rb') as f:
            processor.__dict__.update(pickle.load(f))
        logger.info("Processor state loaded from %s", filepath)
        return processor
    except Exception as e:
        logger.error("Error loading processor state: %s", e)
        return None
