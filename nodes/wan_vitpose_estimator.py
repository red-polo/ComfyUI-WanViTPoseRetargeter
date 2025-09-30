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
        dw_pose_model = "pose2d/vitpose_h_wholebody.onnx"
        yolo_model = "det/yolov10m.onnx"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "..", "models")

        pose2d_checkpoint_path = os.path.join(model_base_path, dw_pose_model)
        det_checkpoint_path = os.path.join(model_base_path, yolo_model)

        pose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)
        frames = image.cpu().numpy() * 255
        refer_img = frames[0].copy()   
        tpl_pose_metas = pose2d(frames)
        tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]
        cond_images = []
        
        for idx, meta in enumerate(tpl_retarget_pose_metas):
            canvas = np.zeros_like(refer_img)
            conditioning_image = draw_aapose_by_meta_new(canvas, meta)
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
        #print(images.shape)
        
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

        tpl_retarget_pose_metas = get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, None, None,
                                                     False, 1.0, "neck",0, 0,calibration_pose_meta=None)
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


class WanViTPoseRetargeterToSrc:
    """WanViTPoseRetargeteToSrc"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "images": ("IMAGE", {"tooltip": "Input image for pose detection"}),
                "ref_image": ("IMAGE", {"tooltip": "Input reference image"}),
                # "target_to_src": ("BOOLEAN", {
                #     "default": False,
                #     "tooltip": "True: scale to src / False: scale to ref"
                # }),
                "adjust_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,       # 最小値
                    "max": 100.0,       # 最大値
                    "step": 0.05,
                    "tooltip": "大きさをスケールする"
                }),
                #hipもやりたい
                "adjust_scale_anker": (["neck","around foot"],{"default": "neck","tooltip": "target_to_srcがTrueの時のみ有効"}),
                
                "adjust_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -2000.0,       # 最小値
                    "max": 2000.0,       # 最大値
                    "step": 1.0,
                    "tooltip": "x方向に移動させる"
                }),
                "adjust_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -2000.0,       # 最小値
                    "max": 2000.0,       # 最大値
                    "step": 1.0,
                    "tooltip": "y方向に移動させる"
                }),
                # "bone_scale_calc_index": ("INT", {
                #     "default": 0,
                #     "min":0,       # 最小値
                #     "max": 2000,       # 最大値
                #     "step": 1.0,
                #     "tooltip": "index"
                # }),
            },
            # "optional": {
            #     "calibration_image": ("IMAGE", {"tooltip": "Input caribration image"}),
            #     #"calibration_ref_image": ("IMAGE", {"tooltip": "Input reference caribration image"})
            # }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cond_images",)
    FUNCTION = "run"
    CATEGORY = "WanViTPoseRetargeter"


    def run(self, images,ref_image , adjust_scale, adjust_scale_anker, adjust_x, adjust_y) :
        calibration_image = None
        keep_src_scale = True
        bone_scale_calc_index=0
        # Model loading
        dw_pose_model = "pose2d/vitpose_h_wholebody.onnx"
        yolo_model = "det/yolov10m.onnx"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "..", "models")

        pose2d_checkpoint_path = os.path.join(model_base_path, dw_pose_model)
        det_checkpoint_path = os.path.join(model_base_path, yolo_model)

        pose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)
        frames = images.cpu().numpy() * 255
        # if calibration_image == None:
        frame0 = frames[bone_scale_calc_index:bone_scale_calc_index+1].copy()   
        # else:
        #     frame0 = calibration_image.cpu().numpy() * 255
        ref_img = ref_image.cpu().numpy() * 255
        
        tpl_pose_metas = pose2d(frames)
        tpl_pose_meta0 = pose2d(frame0)[0]
        refer_pose_meta = pose2d(ref_img)[0]
        calibration_pose_meta=None
        if calibration_image is not None:
            calibration_image = calibration_image.cpu().numpy() * 255
            calibration_pose_meta = pose2d(calibration_image)[0]
        tpl_retarget_pose_metas = get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, None, None,
                                                     keep_src_scale, adjust_scale, adjust_scale_anker,adjust_x, adjust_y,calibration_pose_meta=calibration_pose_meta)

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
