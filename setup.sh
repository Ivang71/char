python3 -m pip install --upgrade pip
export MAKEFLAGS="-j$(nproc)"
pip install numpy torch
pip install huggingface_hub[hf_transfer] hf_transfer sentencepiece transformers accelerate flashinfer-python xformers \
    git+https://github.com/huggingface/diffusers.git --upgrade --break-system-packages

# pip install "flash-attn>=2.7.1,<=2.8.0" --no-build-isolation --break-system-packages


TOKEN=$(echo 'aGZfaHZqck9VTXFvTXF3dW9HR3JoTlZKSWlsZUtFTlNQbXRjTw==' | base64 --decode)
export HUGGINGFACEHUB_API_TOKEN="$TOKEN"
export HF_TOKEN="$TOKEN"
export HF_XET_HIGH_PERFORMANCE=1
huggingface-cli login --token "$TOKEN" --add-to-git-credential=false

huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev
