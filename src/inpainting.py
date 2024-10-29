import torch
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
from PIL import Image, ImageFilter
from src.segmentation_colors import map_colors_rgb
from shared_utils import filter_items
import numpy as np

class Inpainting:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = AutoPipelineForInpainting.from_pretrained('lykon/absolute-reality-1.6525-inpainting', torch_dtype=torch.float16, variant="fp16")
        self.pipe.scheduler = DEISMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

    def get_inpainting_mask(self, segmentation_mask: np.ndarray) -> Image:
        unique_colors = np.unique(segmentation_mask.reshape(-1, segmentation_mask.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]

        control_items = ["windowpane;window", "wall", "floor;flooring","ceiling",  "sconce", "door;double;door", "light;light;source",
                        "painting;picture", "stairs;steps","escalator;moving;staircase;moving;stairway"]
        chosen_colors, segment_items = filter_items(
                    colors_list=unique_colors,
                    items_list=segment_items,
                    items_to_remove=control_items
                )

        mask = np.zeros_like(segmentation_mask)
        for color in chosen_colors:
            color_matches = (segmentation_mask == color).all(axis=2)
            mask[color_matches] = 1

        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        mask_image = mask_image.filter(ImageFilter.MaxFilter(25))
        return mask_image

    def cleanup_room(self, image: Image, mask: Image) -> Image:
        inpaint_prompt = "Empty room, with only empty walls, floor, ceiling, doors, windows"

        new_negative_prompt = "bed, mattress, fence, gate, desk, nightstand, dresser, cabinet, office chair, recliner, ottoman, bench, bar stool, side table, chaise, loveseat, futon, bunk bed, hutch, shoe rack, hall tree, chair"
        negative_prompt = new_negative_prompt + " furniture, sofa, cough, table, plants, rug, home equipment, music equipment, shelves, books, light, lamps, window, radiator"
        image_source_for_inpaint = image.resize((512, 512))
        image_mask_for_inpaint = mask.resize((512, 512))
        generator = [torch.Generator(device="cuda").manual_seed(20)]

        image_inpainting_auto = self.pipe(prompt=inpaint_prompt, negative_prompt=negative_prompt, generator=generator, strentgh=0.8,
            image=image_source_for_inpaint, mask_image=image_mask_for_inpaint, guidance_scale=10.0,
            num_inference_steps=10).images[0]
        image_inpainting_auto = image_inpainting_auto.resize((image.size[0], image.size[1]))
        return image_inpainting_auto