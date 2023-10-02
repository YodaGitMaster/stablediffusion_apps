from diffusers import StableDiffusionPipeline
import torch
import random
import requests
from PIL import Image
import os 
from datetime import datetime
import json

def log_entry(file_name, seed):
    entry = {
        'file_name': file_name,
        'seed': seed
    }
    with open('log.jsonl', 'a') as file:
        file.write(json.dumps(entry) + '\n')

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
generator = torch.Generator(device='cuda')

negative_prompt = ''
prompt = ''

for x in range(6):
    seed = generator.seed()
    generator = generator.manual_seed(seed)
    
    image = pipe(prompt,guidance_scale=8, num_inference_steps=75, negative_prompt=negative_prompt, generator=generator).images[0]
    folder_path = os.getcwd()
    new_folder_path = os.path.join(folder_path, "output")
    os.makedirs(new_folder_path, exist_ok=True)
    time= str(datetime.now().time()).replace(":","_")
    date =  str(datetime.now().date()).replace(":","_")
    name = f"{date}_{time}_photorealism_{x}.png"
    image_path = os.path.join(new_folder_path, name)
    image.save(image_path)
    
    log_entry(name,seed)
    


#https://huggingface.co/blog/stable_diffusion, https://huggingface.co/stabilityai