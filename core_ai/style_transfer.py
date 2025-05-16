# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import random
from typing import Dict
from config import MODELS_DIR  # If config.py exists
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as vgg
from PIL import Image
import os
import logging
from typing import List, Tuple
from config import PROJECT_CONFIG  # For directory structure and settings

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
STYLE_DIR = os.path.join(DATA_DIR, 'styles')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, OUTPUT_DIR, STYLE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VGGFeatures(nn.Module):
    """Extract features from specific VGG19 layers for style transfer."""

    def __init__(self, layer_indices: List[int]):
        super(VGGFeatures, self).__init__()
        self.layer_indices = layer_indices
        vgg_model = vgg.vgg19(pretrained=True).features
        self.layers = nn.ModuleList()
        for i, layer in enumerate(vgg_model):
            self.layers.append(layer)
            if i >= max(layer_indices):
                break
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features


def load_image(
        image_path: str,
        size: int = 512,
        device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Load and preprocess an image for style transfer.

    Args:
        image_path: Path to the image file.
        size: Target size for resizing.
        device: Device to load the tensor onto.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        logger.info(f"Loaded image from {image_path} with size {size}x{size}")
        return img_tensor
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        raise


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix for style loss."""
    batch, channels, height, width = tensor.size()
    features = tensor.view(batch * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch * channels * height * width)


def compute_content_loss(
        target: torch.Tensor,
        content: torch.Tensor) -> torch.Tensor:
    """Compute content loss between target and content features."""
    return torch.mean((target - content) ** 2)


def compute_style_loss(
        target: torch.Tensor,
        style: torch.Tensor) -> torch.Tensor:
    """Compute style loss between target and style features."""
    return torch.mean((gram_matrix(target) - gram_matrix(style)) ** 2)


def style_transfer(
        content_path: str,
        style_path: str,
        output_path: str,
        steps: int = 300,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        lr: float = 0.01) -> None:
    """
    Perform neural style transfer.

    Args:
        content_path: Path to content image.
        style_path: Path to style image.
        output_path: Path to save the styled image.
        steps: Number of optimization steps.
        content_weight: Weight for content loss.
        style_weight: Weight for style loss.
        lr: Learning rate for optimization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = load_image(content_path, device=device)
    style_img = load_image(style_path, device=device)

    # Define VGG layers for content and style
    content_layers = [21]  # conv4_2
    # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    style_layers = [0, 5, 10, 19, 28]
    model = VGGFeatures(style_layers + content_layers).to(device).eval()

    # Extract features
    content_features = model(content_img)
    style_features = model(style_img)
    content_targets = [content_features[content_layers[0]]]
    style_targets = [style_features[i] for i in range(len(style_layers))]

    # Initialize target image
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=lr)

    # Optimization loop
    for step in range(steps):
        optimizer.zero_grad()
        target_features = model(target)

        # Compute losses
        content_loss = sum(
            compute_content_loss(
                target_features[i],
                content_targets[i]) for i in range(
                len(content_layers)))
        style_loss = sum(
            compute_style_loss(
                target_features[i],
                style_targets[i]) for i in range(
                len(style_layers)))
        total_loss = content_weight * content_loss + style_weight * style_loss

        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            logger.info(
                f"Step {step}/{steps}, Content Loss: {
                    content_loss.item():.4f}, " f"Style Loss: {
                    style_loss.item():.4f}, Total Loss: {
                    total_loss.item():.4f}")

    # Save result
    save_image(target, output_path)
    logger.info(f"Styled image saved to {output_path}")


def save_image(tensor: torch.Tensor, output_path: str) -> None:
    """Save a tensor as an image."""
    try:
        tensor = tensor.cpu().detach().squeeze(0)
        tensor = torch.clamp(tensor, 0, 1)  # Ensure values are in [0, 1]
        transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.ToPILImage()
        ])
        img = transform(tensor)
        img.save(output_path)
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        raise


def load_raw_images() -> Dict[str, torch.Tensor]:
    """Load images from data/raw for style transfer."""
    raw_images = {}
    size = PROJECT_CONFIG.get("style_transfer", {}).get("image_size", 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(RAW_DIR, filename)
            try:
                img = load_image(filepath, size=size, device=device)
                raw_images[filename] = img
            except Exception as e:
                logger.error(f"Failed to load raw image {filepath}: {e}")
    return raw_images


def batch_style_transfer(
        content_paths: List[str],
        style_path: str,
        output_prefix: str = "styled_") -> None:
    """Perform style transfer on multiple content images."""
    for content_path in content_paths:
        output_path = os.path.join(
            OUTPUT_DIR, f"{output_prefix}{
                os.path.basename(content_path)}")
        style_transfer(content_path, style_path, output_path)


def dangerous_style_transfer(
        content_path: str,
        style_path: str,
        output_path: str,
        steps: int = 300) -> None:
    """Perform style transfer with risky optimization (dangerous AI theme)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = load_image(content_path, device=device)
    style_img = load_image(style_path, device=device)

    model = VGGFeatures([21, 0, 5, 10, 19, 28]).to(device).eval()
    content_features = model(content_img)[0]
    style_features = model(style_img)[1:]

    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.SGD([target], lr=0.1)  # High learning rate

    for step in range(steps):
        optimizer.zero_grad()
        target_features = model(target)

        content_loss = compute_content_loss(
            target_features[0], content_features)
        style_loss = sum(compute_style_loss(target_features[i + 1], style_features[i])
                         for i in range(len(style_features)))
        total_loss = content_loss + style_loss * \
            random.uniform(1e6, 1e8)  # Risky style weight

        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            logger.warning(
                f"Dangerous Step {step}/{steps}, Loss: {total_loss.item():.4f}")

    save_image(target, output_path)
    logger.info(f"Dangerous styled image saved to {output_path}")


def save_style_model(model: nn.Module, filename: str) -> bool:
    """Save the style transfer model state."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        torch.save(model.state_dict(), filepath)
        logger.info(f"Style model saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving style model to {filepath}: {e}")
        return False


def load_style_model(model: nn.Module, filename: str) -> bool:
    """Load the style transfer model state."""
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"Style model file {filepath} not found")
        return False

    try:
        model.load_state_dict(torch.load(filepath))
        model.eval()
        logger.info(f"Style model loaded from {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error loading style model from {filepath}: {e}")
        return False


def main():
    """Demonstrate style transfer functionality."""
    content_path = os.path.join(
        RAW_DIR, "content.jpg")  # Placeholder; replace with actual path
    style_path = os.path.join(STYLE_DIR, "boondocks_style.jpg")
    output_path = os.path.join(OUTPUT_DIR, "output_styled_image.jpg")

    # Standard style transfer
    logger.info("Running standard style transfer...")
    style_transfer(content_path, style_path, output_path, steps=300)

    # Batch style transfer with raw images
    logger.info("Running batch style transfer with raw images...")
    raw_images = load_raw_images()
    content_paths = [os.path.join(RAW_DIR, name) for name in raw_images.keys()]
    if content_paths:
        batch_style_transfer(content_paths, style_path)

    # Dangerous mode demo
    logger.info("Running dangerous style transfer...")
    dangerous_output = os.path.join(OUTPUT_DIR, "dangerous_styled_image.jpg")
    dangerous_style_transfer(content_path, style_path, dangerous_output)


if __name__ == "__main__":
    main()

# Additional utilities


def preprocess_image_batch(
        image_paths: List[str], size: int = 512) -> List[torch.Tensor]:
    """Preprocess a batch of images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [load_image(path, size=size, device=device) for path in image_paths]


def evaluate_style_transfer(content_img: torch.Tensor, style_img: torch.Tensor,
                            target_img: torch.Tensor) -> Dict[str, float]:
    """Evaluate the quality of style transfer."""
    model = VGGFeatures([21, 0, 5, 10, 19, 28]).to(content_img.device).eval()
    content_features = model(content_img)[0]
    style_features = model(style_img)[1:]
    target_features = model(target_img)

    content_loss = compute_content_loss(
        target_features[0], content_features).item()
    style_loss = sum(compute_style_loss(target_features[i + 1], style_features[i]).item()
                     for i in range(len(style_features)))

    return {'content_loss': content_loss, 'style_loss': style_loss}


def optimize_style_weights(
        content_path: str, style_path: str, steps: int = 100) -> Tuple[float, float]:
    """Optimize content and style weights experimentally."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = load_image(content_path, device=device)
    style_img = load_image(style_path, device=device)

    best_weights = (1.0, 1e6)
    best_loss = float('inf')

    for cw in [0.5, 1.0, 2.0]:
        for sw in [1e5, 1e6, 1e7]:
            target = content_img.clone().requires_grad_(True).to(device)
            optimizer = optim.Adam([target], lr=0.01)
            model = VGGFeatures([21, 0, 5, 10, 19, 28]).to(device).eval()

            for _ in range(steps):
                optimizer.zero_grad()
                target_features = model(target)
                content_loss = compute_content_loss(
                    target_features[0], model(content_img)[0])
                style_loss = sum(compute_style_loss(
                    target_features[i + 1], model(style_img)[i + 1]) for i in range(5))
                total_loss = cw * content_loss + sw * style_loss
                total_loss.backward()
                optimizer.step()

            loss_value = total_loss.item()
            if loss_value < best_loss:
                best_loss = loss_value
                best_weights = (cw, sw)

    logger.info(
        f"Optimized weights: content={
            best_weights[0]}, style={
            best_weights[1]}")
    return best_weights
