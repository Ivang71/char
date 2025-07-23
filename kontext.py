import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("https://i.pinimg.com/1200x/30/c7/c0/30c7c096959245ce68c47faa4302c4bf.jpg")

image = pipe(
  image=input_image,
  prompt="Add her a hat",
  guidance_scale=2.5
).images[0]

image.save("output.jpg")

print("Image saved successfully")