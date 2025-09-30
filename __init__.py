"""ComfyUI-WanViTPoseEstimator
"""
from .nodes.wan_vitpose_estimator import WanViTPoseEstimator,WanViTPoseRetargeter,WanViTPoseRetargeterToSrc

__all__ = ["WanViTPoseEstimator","WanViTPoseRetargeter"]
__version__ = "0.1.0"

# ComfyUI requires these mappings at module import time.
NODE_CLASS_MAPPINGS = {
    "WanViTPoseEstimator": WanViTPoseEstimator,
    "WanViTPoseRetargeter": WanViTPoseRetargeter,
    "WanViTPoseRetargeterToSrc": WanViTPoseRetargeterToSrc,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanViTPoseEstimator": "WanViTPoseEstimator",
    "WanViTPoseRetargeter": "WanViTPoseRetargeter",
    "WanViTPoseRetargeterToSrc": "WanViTPoseRetargeterToSrc",
}
