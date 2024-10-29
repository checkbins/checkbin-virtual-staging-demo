import torch
from diffusers.pipelines.controlnet import StableDiffusionControlNetInpaintPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLPipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
from src.colors import ade_palette
from shared_utils import resize_dimensions, flush, filter_items, map_colors_rgb
from segmentation import Segmentation
from depth_estimation import DepthEstimation

class ControlNetDepthDesignModelMulti:
    """ Produces random noise images """
    
    def __init__(self):
        """ Initialize your model(s) here """
        self.device = "cuda"
        self.dtype = torch.float16
        self.seed = 323*111
        self.neg_prompt = "window, door, low resolution, banner, logo, watermark, text, deformed, blurry, out of focus, surreal, ugly, beginner"
        self.control_items = ["windowpane;window", "door;double;door"]
        self.additional_quality_suffix = "interior design, 4K, high resolution, photorealistic"

        controlnet_depth = ControlNetModel.from_pretrained(
            "timlenardo/controlnet_depth", torch_dtype=self.dtype, use_safetensors=True)
        controlnet_seg = ControlNetModel.from_pretrained(
            "timlenardo/own_controlnet", torch_dtype=self.dtype, use_safetensors=True)

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            controlnet=[controlnet_depth, controlnet_seg],
            safety_checker=None,
            torch_dtype=self.dtype
        )

        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                            weight_name="ip-adapter_sd15.bin")
        self.pipe.set_ip_adapter_scale(0.4)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        self.guide_pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B",
                                                            torch_dtype=self.dtype, use_safetensors=True, variant="fp16")
        self.guide_pipe = self.guide_pipe.to(self.device)
        
        self.segmentation = Segmentation(model_type='upernet')
        self.depth_estimator = DepthEstimation()
        


    def generate_design(self, empty_room_image: Image, prompt: str, guidance_scale: int = 10, num_steps: int = 50, strength: float =0.9, img_size: int = 640) -> Image:
        """
        Given an image of an empty room and a prompt
        generate the designed room according to the prompt
        Inputs - 
            empty_room_image - An RGB PIL Image of the empty room
            prompt - Text describing the target design elements of the room
        Returns - 
            design_image - PIL Image of the same size as the empty room image
                           If the size is not the same the submission will fail.
        """
        print(prompt)
        flush()
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        pos_prompt = prompt + f', {self.additional_quality_suffix}'

        orig_w, orig_h = empty_room_image.size
        new_width, new_height = resize_dimensions(empty_room_image.size, img_size)
        input_image = empty_room_image.resize((new_width, new_height))
        color_seg, seg_image = self.segmentation.segment_image(input_image)
        real_seg = color_seg # np.array(self.segmentation.segment_image(input_image))
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]
        chosen_colors, segment_items = filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=self.control_items
        )
        mask = np.zeros_like(real_seg)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1

        image_np = np.array(input_image)
        image = Image.fromarray(image_np).convert("RGB")
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")

        image_depth = self.depth_estimator.get_depth_image(image)

        # generate image that would be used as IP-adapter
        flush()
        new_width_ip = int(new_width / 8) * 8
        new_height_ip = int(new_height / 8) * 8
        ip_image = self.guide_pipe(pos_prompt,
                                   num_inference_steps=num_steps,
                                   negative_prompt=self.neg_prompt,
                                   height=new_height_ip,
                                   width=new_width_ip,
                                   generator=[self.generator]).images[0]

        flush()
        generated_image = self.pipe(
            prompt=pos_prompt,
            negative_prompt=self.neg_prompt,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=[self.generator],
            image=image,
            mask_image=mask_image,
            ip_adapter_image=ip_image,
            control_image=[image_depth, segmentation_cond_image],
            controlnet_conditioning_scale=[0.5, 0.5]
        ).images[0]
        
        flush()
        design_image = generated_image.resize(
            (orig_w, orig_h), Image.Resampling.LANCZOS
        )
        
        return design_image