import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import time

# 1. Load pipeline with bfloat16 and Flash Attention 2
# attn_implementation="flash_attention_2" speeds up attention and saves memory.
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
pipe.to("cuda")

# 2. Apply torch.compile for a significant speed boost
# The 'max-autotune' mode provides the best performance after a one-time
# compilation overhead on the first inference run.
print("Compiling the transformer model...")
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
print("Compilation complete.")

input_image = load_image("https://i.pinimg.com/1200x/30/c7/c0/30c7c096959245ce68c47faa4302c4bf.jpg")

# 3. Run inference with guidance_scale=0.0
# FLUX models work well with zero guidance. This skips an extra model pass,
# making inference faster.
print("Running inference...")
start_time = time.time()
image = pipe(
    image=input_image,
    prompt="Add her a hat",
    guidance_scale=0.0
).images[0]
end_time = time.time()
print(f"Inference took {end_time - start_time:.2f} seconds.")

image.save("output.jpg")
print("Image saved successfully")z