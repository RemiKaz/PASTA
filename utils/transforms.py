import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

import utils


class AddInverse(torch.nn.Module):
    """To a [B, C, H, W] input add the inverse channels of the given one to it.
    Results in a [B, 2C, H, W] output. Single image [C, H, W] is also accepted.

    Args:
        dim (int): where to add channels to. Default: -3
    """

    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: Tensor) -> Tensor:
        return torch.cat([in_tensor, 1 - in_tensor], dim=self.dim)


class BrightnessTransform(torchvision.transforms.Lambda):
    def __init__(self, magnitude):
        super().__init__(lambda img: F.adjust_brightness(img, 1.0 + magnitude))


class ColorTransform(torchvision.transforms.Lambda):
    def __init__(self, magnitude):
        super().__init__(lambda img: F.adjust_saturation(img, 1.0 + magnitude))


class ContrastTransform(torchvision.transforms.Lambda):
    def __init__(self, magnitude):
        super().__init__(lambda img: F.adjust_contrast(img, 1.0 + magnitude))


class SharpnessTransform(torchvision.transforms.Lambda):
    def __init__(self, magnitude):
        super().__init__(lambda img: F.adjust_sharpness(img, 1.0 + magnitude))


class PosterizeTransform(torchvision.transforms.Lambda):
    def __init__(self, magnitude):
        super().__init__(lambda img: F.posterize(img, int(magnitude)))


class SolarizeTransform(torchvision.transforms.Lambda):
    def __init__(self, magnitude):
        super().__init__(lambda img: F.solarize(img, magnitude))


class ApplyMasksTransform(torchvision.transforms.Lambda):
    """Applies a given number of masks to an image."""

    def __init__(self, magnitude):
        """Initializes the ApplyMasksTransform.

        Args:
            magnitude (int): The number of masks to apply.
        """
        super().__init__(self.apply_masks)
        self.magnitude = magnitude

    def apply_masks(self, img):
        """Applies masks to the image.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The image with masks applied.
        """
        masks = int(self.magnitude)
        for _i in range(masks):
            # Create a mask with all pixels set to 0
            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            # Generate a random rectangle
            rect = utils.randomize_rectangle(img.size, 224, 224)
            # Draw the rectangle on the mask
            draw.rectangle(rect, fill=255)
            # Apply the mask to the image
            img = utils.mask_image(img, mask)
        return img


class ImagePerturbator:
    """Class for perturbating an image."""

    def __init__(self):
        """Initializes the image perturbator."""
        self.perturbation_methods = [
            "ColorJitter_brightness",
            "ColorJitter_contrast",
            "RandomResizedCrop",
            "GaussianBlur",
            "RandomPerspective",
            "BrightnessTransform",
            "ColorTransform",
            "ContrastTransform",
            "SharpnessTransform",
            "PosterizeTransform",
            "SolarizeTransform",
            "ApplyMasksTransform",
        ]

    def perturbate_image(self, image, id_perturbation, magnitude, fix_seed=False):
        """Perturbates an image.

        Args:
            image (PIL.Image.Image): The input image.
            id_perturbation (int): The index of the perturbation method.
            magnitude (float): The magnitude of the perturbation.
            fix_seed (bool): Whether to fix the seed of the random number generator.

        Returns:
            PIL.Image.Image: The perturbed image.
        """
        if fix_seed:
            seed = 493
            random.seed(seed)
            torch.manual_seed(seed)

        # Apply the selected perturbation method
        if self.perturbation_methods[id_perturbation] == "ColorJitter_brightness":
            # Adjust the brightness of the image
            return transforms.ColorJitter(brightness=np.tanh(magnitude * 0.5))(image)
        if self.perturbation_methods[id_perturbation] == "ColorJitter_contrast":
            # Adjust the contrast of the image
            return transforms.ColorJitter(contrast=np.tanh(magnitude * 0.5))(image)
        if self.perturbation_methods[id_perturbation] == "RandomResizedCrop":
            # Randomly resize and crop the image
            return transforms.RandomResizedCrop(
                224,
                scale=(1 - np.tanh(magnitude * 0.3), 1 - np.tanh(magnitude * 0.3)),
                antialias=True,
            )(image)
        if self.perturbation_methods[id_perturbation] == "GaussianBlur":
            # Apply gaussian blur to the image
            return transforms.GaussianBlur(kernel_size=15, sigma=2 * (magnitude + 0.00001))(image)
        if self.perturbation_methods[id_perturbation] == "RandomPerspective":
            # Randomly apply perspective transformation to the image
            return transforms.RandomPerspective(distortion_scale=0.5 * magnitude, p=1)(image)
        if self.perturbation_methods[id_perturbation] == "BrightnessTransform":
            # Apply brightness transformation to the image
            return BrightnessTransform(0.5 * magnitude)(image)
        if self.perturbation_methods[id_perturbation] == "ColorTransform":
            # Apply color transformation to the image
            return ColorTransform(2 * magnitude)(image)
        if self.perturbation_methods[id_perturbation] == "ContrastTransform":
            # Apply contrast transformation to the image
            return ContrastTransform(magnitude)(image)
        if self.perturbation_methods[id_perturbation] == "SharpnessTransform":
            # Apply sharpness transformation to the image
            return SharpnessTransform(5 * magnitude)(image)
        if self.perturbation_methods[id_perturbation] == "PosterizeTransform":
            # Apply posterize transformation to the image
            return PosterizeTransform(8 - np.min([8, magnitude + 1]))(image)
        if self.perturbation_methods[id_perturbation] == "SolarizeTransform":
            # Apply solarize transformation to the image
            return SolarizeTransform(256 - np.min([256, magnitude * 5]))(image)
        if self.perturbation_methods[id_perturbation] == "ApplyMasksTransform":
            # Apply masks transformation to the image
            return ApplyMasksTransform(magnitude)(image)
        return None

    def mixup_perturbation(self, img_base, img_perturb, magnitude):
        img_base_array = np.array(img_base)
        img_perturb_array = np.array(img_perturb)
        mixed_image_array = (
            (1 - magnitude) * img_base_array + magnitude * img_perturb_array
        ).astype(np.uint8)
        return Image.fromarray(mixed_image_array)
