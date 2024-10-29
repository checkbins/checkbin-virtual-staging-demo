from modal import App, gpu, Image as ModalImage, Mount, Secret
from src.main import run_test
from modal import (
    Image as ModalImage
)
import os,sys


remove_furniture_image = (
    ModalImage.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg", "libsm6", "libxext6"])
    .pip_install(
        [
            "torch==2.4.0", # This doesn't seem to work, torch is 2.1.2 when printed later? 
            "torchvision",
        ]
    )
    .pip_install(
        [
            "diffusers",
            "xformers==0.0.23.post1",
            "transformers==4.39.1",
            "accelerate==0.26.1",
            "opencv-python==4.9.0.80",
            "scipy==1.11.4",
            "triton==2.1.0",
            "altair==4.1.0",
            "pandas==2.1.4",
            "requests",
            "azure-storage-blob",
            "azure-storage-file-datalake",
            "tqdm",
            "typed-argument-parser",
            "boto3",
            "google-cloud-storage",
            "huggingface-hub",
        ]
    )
    .run_commands(
        ["pip show torch", "pip3 install natten==0.17.1+torch210cu121 -f https://shi-labs.com/natten/wheels/"]
    )
)

app = App("virtual-staging", image=remove_furniture_image)

@app.function(
    gpu=gpu.A10G(),
    timeout=3600,
    image=remove_furniture_image,
    secrets=[Secret.from_name("checkbin-secret")],
    mounts=[Mount.from_local_dir("./checkbin-python", remote_path="/root/checkbin-python")],

)
def run_batch():
    # TODO: move to pip package after finalized. 
    sys.path.insert(0, 'checkbin-python/src')
    import checkbin

    checkbin.authenticate(token=os.environ["CHECKBIN_TOKEN"])
    checkbin_app = checkbin.App(app_key="virtual_staging_demo", mode="remote")

    # TODO: uncomment this to create a new input set from code!
    #checkbin_input_set = checkbin_app.create_input_set("20x Test Inputs")
    #input_set_id = checkbin_input_set.create_from_json(json_file='./test_inputs.json')
    # print(input_set_id) # Store the input_set_id to use in the 

    # TODO: replace this with the input set ID you create!
    input_set_id = "bcbfb43d-c194-435d-8dba-b21b7705149e" 

    # with checkbins = checkbin_app.start_run(set_id="0a90b9ab-ea7a-40c3-aea4-addc71b1b62b") as checkbins: # This is the subset of images where the furniture is not correctly removed.
    with checkbin_app.start_run(set_id=input_set_id) as checkbins: # This is the full test set. 
        for checkbin in checkbins: 
            for key, value in checkbin.input_state.items():
                print(f"{key}: {value}")
            image_url = checkbin.input_state["url"]["url"]
            image_prompt = checkbin.input_state["prompt"]["data"]
            print(image_url)
            run_test(image_url=image_url, image_prompt=image_prompt, checkbin=checkbin)
