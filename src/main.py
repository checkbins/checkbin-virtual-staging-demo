from PIL import Image
import requests
from io import BytesIO
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from segmentation import Segmentation
from inpainting import Inpainting
from depth_estimation import DepthEstimation
from model import ControlNetDepthDesignModelMulti
import numpy as np

def run_test(image_url, image_prompt, checkbin):
    segmentation = Segmentation()
    inpainting = Inpainting()
    depth_estimation = DepthEstimation()

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))    
    image_id = os.path.basename(image_url).split(".")[0]

    # Step 1 - Segment Image
    color_map, segmentation_map = segmentation.segment_image(image)
    inpaniting_mask = inpainting.get_inpainting_mask(color_map)

    # Check the array properties
    segmentation_map_array = np.array(segmentation_map)
    print(segmentation_map_array.shape)  # e.g., (height, width, 3) for an RGB image
    print(segmentation_map_array.dtype)  # e.g., dtype('uint8')
    print(segmentation_map_array.min(), segmentation_map_array.max())  # Should be in the range 0-255

    checkbin.checkin(name="segmentation")
    checkbin.upload_array_as_image(name="segmentation_map", array=np.array(segmentation_map))
    checkbin.upload_array_as_image(name="inpainting_mask", array=np.array(inpaniting_mask))

    # Step 2 - Remove furniture
    clean_room = inpainting.cleanup_room(image, inpaniting_mask)
    color_map_clean, segmentation_map_clean_room = segmentation.segment_image(clean_room)
    depth_clean_room = depth_estimation.get_depth_image(clean_room)

    checkbin.checkin(name="cleaned")
    checkbin.upload_array_as_image(name="clean_room", array=np.array(clean_room))
    checkbin.upload_array_as_image(name="segmentation_map_clean_room", array=np.array(segmentation_map_clean_room))
    checkbin.upload_array_as_image(name="depth_clean_room", array=np.array(depth_clean_room))

    # Step 3 - Add new furniture to match prompt
    generativeModel = ControlNetDepthDesignModelMulti()
    generated_image = generativeModel.generate_design(clean_room, image_prompt)

    checkbin.checkin(name="generated")
    checkbin.upload_array_as_image(name="generated_image", array=np.array(generated_image))

    checkbin.submit()
