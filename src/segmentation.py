import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
from typing import Tuple
from src.segmentation_colors import ade_palette
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

class Segmentation:
    def __init__(self, model_type="oneformer", device="cuda"):
        self.device = device
        self.model_type = model_type
        if (model_type == "oneformer"):
            self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
            self.model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
        elif (model_type == "upernet"):
            self.processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
            self.model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        else: 
            raise ValueError("unknown model type")

        self.model = self.model.to(self.device)

    def segment_image(self, image: Image) -> tuple[np.ndarray, Image]:
        if self.model_type == "oneformer":
            return self.segment_image_oneformer(image=image)
        elif self.model_type == "upernet":
            return self.segment_image_upernet(image=image)
        else: 
            raise ValueError("unknown model type")

    def segment_image_oneformer(self, image: Image) -> tuple[np.ndarray, Image]:
        with torch.inference_mode():
            semantic_inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
            semantic_inputs = {key: value.to(self.device) for key, value in semantic_inputs.items()}
            semantic_outputs = self.model(**semantic_inputs)
            predicted_semantic_map = self.processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]

        predicted_semantic_map = predicted_semantic_map.cpu()
        color_seg = np.zeros((predicted_semantic_map.shape[0], predicted_semantic_map.shape[1], 3), dtype=np.uint8)

        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[predicted_semantic_map == label, :] = color

        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert('RGB')
        return color_seg, seg_image
    
    
    def segment_image_upernet(
        self,
        image: Image,
    ) -> tuple[np.ndarray, Image]:
        """
        Segments an image using a semantic segmentation model.

        Args:
            image (Image): The input image to be segmented.
            image_processor (AutoImageProcessor): The processor to prepare the
                image for segmentation.
            image_segmentor (UperNetForSemanticSegmentation): The semantic
                segmentation model used to identify different segments in the image.

        Returns:
            Image: The segmented image with each segment colored differently based
                on its identified class.
        """
        # image_processor, image_segmentor = get_segmentation_pipeline()
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.model(pixel_values.to(self.device))

        seg = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        seg_cpu = seg.cpu()
        for label, color in enumerate(palette):
            color_seg[seg_cpu == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert('RGB')
        return color_seg, seg_image