from diffusers import StableDiffusionPipeline,StableDiffusionUpscalePipeline
import torch
import random
import requests
from PIL import Image
import os 
from datetime import datetime
random_number = random.randint(1000, 9999)


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if torch.cuda.is_available():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
     prompt='create a photo reppresenting: water wars in 2030'
    negative_prompt = 'aliens'
    
    folder_path = os.getcwd()
    new_folder_path = os.path.join(folder_path, "output3")
    os.makedirs(new_folder_path, exist_ok=True)
    for x in range(10):
        image = pipe(prompt,height=512,width=512,guidance_scale=16, num_inference_steps=50, negative_prompt=negative_prompt).images[0]
        # random_number = random.randint(1, 9999)
        time= str(datetime.now().time()).replace(":","_")
        image_path = os.path.join(new_folder_path, f"{time}_{x}.png")
        image.save(image_path)


else:
    print(torch.cuda.is_available())
    
