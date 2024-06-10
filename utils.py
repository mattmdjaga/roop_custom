from typing import Dict, List, Any, Tuple
import io
import base64
import sys
import os
#sys.path.append("/home/azureuser/API_script/roop_custom/CodeFormer/CodeFormer")

codeformer_dir = os.path.dirname(__file__) + "/CodeFormer/CodeFormer" 
sys.path.append(codeformer_dir)

import cv2
import torch
import onnxruntime
import torch.nn.functional as F
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer

from basicsr.utils.registry import ARCH_REGISTRY


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_codeformer_models():
    upsampler = set_realesrgan()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt_path = "roop_custom/CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
    checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

    return upsampler, codeformer_net, device

# set enhancer with RealESRGAN
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="roop_custom/CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

def encode_execution_providers(execution_providers):
    return [
        execution_provider.replace("ExecutionProvider", "").lower()
        for execution_provider in execution_providers
    ]

def decode_execution_providers(execution_providers):
    return [
        provider
        for provider, encoded_execution_provider in zip(
            onnxruntime.get_available_providers(),
            encode_execution_providers(onnxruntime.get_available_providers()),
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in execution_providers
        )
    ]


def suggest_execution_providers():
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return 8
    return 1


def update_status(message: str, scope: str = "ROOP.CORE") -> None:
    print(f"[{scope}] {message}")
    if not roop.globals.headless:
        ui.update_status(message)

def inference_codeformer(args):
    """Run a single prediction on the model"""
    (
        image,
        face_align,
        background_enhance,
        face_upsample,
        upscale,
        codeformer_fidelity,
        upsampler,
        codeformer_net,
        device,
    ) = args
    try:  # global try
        # take the default setting for the demo
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        # print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        has_aligned = not face_align
        upscale = 1 if has_aligned else upscale

        # img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print('\timage size:', img.shape)

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None
        face_helper.read_image(img)
        # get face landmarks for each face
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        # print(f'\tdetect {num_det_faces} faces')
        # align and warp each face
        face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                # print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        bg_img = None
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img, draw_box=draw_box
        )

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print("Global exception", error)
        return None, None