import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

class DepthEstimation:
    def __init__(self, device="cuda"):
        self.device = device
        self.depth_image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf", torch_dtype=torch.float16)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf", torch_dtype=torch.float16)
        self.depth_model = self.depth_model.to(self.device)

    def get_depth_image(self, image: Image) -> Image:
        image_to_depth = self.depth_image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            depth_map = self.depth_model(**image_to_depth).predicted_depth

        width, height = image.size
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1).float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image