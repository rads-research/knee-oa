"""
Artifact-inspired perturbation patterns.

Four patterns parameterized by a severity multiplier (alpha):
  alpha=0.5 -> 50% severity
  alpha=1.0 -> 100% severity (base parameters)
  alpha=1.5 -> 150% severity

All parameters scale linearly with alpha. Patterns are applied AFTER
preprocessing on normalized [-1, 1] tensors.
"""

import random
import torch


class ArtifactPatterns:
    """Four artifact-inspired perturbation patterns."""

    @staticmethod
    def clean(img, alpha=1.0):
        """No modification (baseline)."""
        return img.clone()

    @staticmethod
    def horizontal_lines(img, alpha=1.0):
        """
        Horizontal gray lines mimicking anti-scatter grid artifacts.

        Base (alpha=1.0): 10px thick lines at intensity=0.5, every 30px,
        starting at row 20.
        """
        out = img.clone()
        base_thickness = 10
        base_intensity = 0.5

        thickness = max(1, int(base_thickness * alpha))
        intensity = base_intensity * alpha

        for row in range(20, 224, 30):
            end_row = min(row + thickness, 224)
            out[:, row:end_row, :] = intensity

        return torch.clamp(out, -1, 1)

    @staticmethod
    def checkerboard(img, alpha=1.0):
        """
        Checkerboard intensity modulation simulating digital processing artifacts.

        Base (alpha=1.0): 16x16 blocks, even blocks +0.3, odd blocks -0.15.
        """
        out = img.clone()
        h, w = out.shape[1], out.shape[2]
        block_size = 16
        base_intensity = 0.3

        intensity = base_intensity * alpha

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    out[:, i:i + block_size, j:j + block_size] += intensity
                else:
                    out[:, i:i + block_size, j:j + block_size] -= intensity * 0.5

        return torch.clamp(out, -1, 1)

    @staticmethod
    def black_bar(img, alpha=1.0):
        """
        Dark bars along margins mimicking PHI redaction or burned-in annotations.

        Base (alpha=1.0): 25px top/bottom bars, 15px left bar, value=-1.0.
        Bar widths scale with alpha, capped at 100px (top/bottom) and 80px (side).
        """
        out = img.clone()
        base_top_bottom = 25
        base_side = 15

        tb = max(1, int(base_top_bottom * alpha))
        side = max(1, int(base_side * alpha))

        tb = min(tb, 100)
        side = min(side, 80)

        out[:, 0:tb, :] = -1.0
        out[:, -tb:, :] = -1.0
        out[:, :, 0:side] = -1.0

        return out

    @staticmethod
    def grid_overlay(img, alpha=1.0):
        """
        Orthogonal grid lines simulating measurement/calibration overlays.

        Base (alpha=1.0): 2px lines every 20px, intensity=0.6.
        """
        out = img.clone()
        h, w = out.shape[1], out.shape[2]
        base_thickness = 2
        base_intensity = 0.6

        thickness = max(1, int(base_thickness * alpha))
        intensity = base_intensity * alpha

        for col in range(0, w, 20):
            end_col = min(col + thickness, w)
            out[:, :, col:end_col] = intensity

        for row in range(0, h, 20):
            end_row = min(row + thickness, h)
            out[:, row:end_row, :] = intensity

        return torch.clamp(out, -1, 1)

    @staticmethod
    def apply_pattern(img, pattern_name, alpha=1.0):
        """Apply named pattern at given severity to an image tensor."""
        patterns = {
            "clean": ArtifactPatterns.clean,
            "horizontal_lines": ArtifactPatterns.horizontal_lines,
            "checkerboard": ArtifactPatterns.checkerboard,
            "black_bar": ArtifactPatterns.black_bar,
            "grid_overlay": ArtifactPatterns.grid_overlay,
        }
        return patterns[pattern_name](img, alpha=alpha)

    @staticmethod
    def apply_random_augmentation(img, prob=0.4, severity_range=(0.5, 1.5)):
        """
        Random artifact augmentation for training.

        With probability `prob`, applies a randomly selected pattern at a
        random severity within `severity_range`.

        Used ONLY during training of augmentation-based (E1) models.
        NOT applied to validation or test images.
        """
        if random.random() > prob:
            return img

        pattern = random.choice([
            "horizontal_lines", "checkerboard", "black_bar", "grid_overlay"
        ])
        alpha = random.uniform(*severity_range)
        return ArtifactPatterns.apply_pattern(img, pattern, alpha=alpha)


# Pattern and severity configuration
PATTERN_NAMES = ["clean", "horizontal_lines", "checkerboard", "black_bar", "grid_overlay"]
SEVERITY_LEVELS = [0.5, 1.0, 1.5]
