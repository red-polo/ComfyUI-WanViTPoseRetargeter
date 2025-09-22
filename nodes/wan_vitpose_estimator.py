from __future__ import annotations
from typing import Any, Dict, Tuple
from .retarget_pose import get_retarget_pose
from .pose2d_utils import AAPoseMeta
from .pose2d import Pose2d
from .human_visualization import draw_aapose_by_meta_new
import numpy as np

import os
import torch
import comfy.model_management as mm


class WanViTPoseEstimator:
    """WanVitPoseEstimator"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image": ("IMAGE", {"tooltip": "Input image for pose detection"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION = "run"
    CATEGORY = "WanViTPoseRetargeter"


    def run(self, image) :
        print(image.shape)
        dw_pose_model = "pose2d/vitpose_h_wholebody.onnx"
        yolo_model = "det/yolov10m.onnx"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "..", "models")

        pose2d_checkpoint_path = os.path.join(model_base_path, dw_pose_model)
        det_checkpoint_path = os.path.join(model_base_path, yolo_model)

        pose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)
        frames = image.cpu().numpy() * 255
        refer_img = frames[0].copy()   
        print(refer_img.shape)
        print(refer_img.dtype)
        print(np.max(refer_img))
        tpl_pose_metas = pose2d(frames)
        print(tpl_pose_metas)
        tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]
        cond_images = []
        
        for idx, meta in enumerate(tpl_retarget_pose_metas):
            canvas = np.zeros_like(refer_img)
            print(canvas.shape)
            print(canvas.dtype)
            conditioning_image = draw_aapose_by_meta_new(canvas, meta)
            print(np.max(conditioning_image))
            cond_images.append(conditioning_image)
        cond_images = np.stack(cond_images, axis=0) / 255

        return  (torch.from_numpy(cond_images),)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> Any:
        return (
            kwargs.get("image", "")
        )

    # @classmethod
    # def VALIDATE_INPUTS(cls, **kwargs) -> bool:
    #     txt = kwargs.get("text", "")
    #     if len(txt) > 10000:
    #         raise ValueError("'text' is too long (max 10,000 chars)." )
    #     return True

class WanViTPoseRetargeter:
    """WanViTPoseRetargeter"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "images": ("IMAGE", {"tooltip": "Input image for pose detection"}),
                "ref_image": ("IMAGE", {"tooltip": "Input reference image"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cond_images",)
    FUNCTION = "run"
    CATEGORY = "WanViTPoseRetargeter"


    def run(self, images,ref_image) :
        print(images.shape)
        
        # Model loading
        dw_pose_model = "pose2d/vitpose_h_wholebody.onnx"
        yolo_model = "det/yolov10m.onnx"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "..", "models")

        pose2d_checkpoint_path = os.path.join(model_base_path, dw_pose_model)
        det_checkpoint_path = os.path.join(model_base_path, yolo_model)

        pose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)
        frames = images.cpu().numpy() * 255
        frame0 = frames[:1].copy()   
        ref_img = ref_image.cpu().numpy() * 255
        
        tpl_pose_metas = pose2d(frames)
        tpl_pose_meta0 = pose2d(frame0)[0]
        refer_pose_meta = pose2d(ref_img)[0]

        tpl_retarget_pose_metas = get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, None, None)
        cond_images = []
        
        for idx, meta in enumerate(tpl_retarget_pose_metas):
            canvas = np.zeros_like(ref_img[0].copy())
            conditioning_image = draw_aapose_by_meta_new(canvas, meta)
            cond_images.append(conditioning_image)
        cond_images = np.stack(cond_images, axis=0)/255

        return  (torch.from_numpy(cond_images),)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> Any:
        return (
            kwargs.get("images", ""),
            kwargs.get("ref_image", ""),
        )

    # @classmethod
    # def VALIDATE_INPUTS(cls, **kwargs) -> bool:
    #     txt = kwargs.get("text", "")
    #     if len(txt) > 10000:
    #         raise ValueError("'text' is too long (max 10,000 chars)." )
    #     return True
