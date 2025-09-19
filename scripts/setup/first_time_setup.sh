
# setup tasks
sudo snap install --classic astral-uv
uv init --python 3.11   # warning: have you already init'd to an earlier python version

uv add omegaconf

# uv venv  # if you don't already have a venv
source .venv/bin/activate

# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# uv add torch torchvision torchaudio
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv add torch
# more info and potentially clean approach:
    # https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch


# git clone https://github.com/state-spaces/mamba.git
# uv pip install mamba/.
# scripts/test_mamba_installation.sh


# git clone https://github.com/goombalab/hnet
# uv pip install -e hnet/.  # initially failed
# #  Failed to build `flash-attn==2.8.0.post2`
# # to resolve I tried to add
# #   [tool.uv.extra-build-dependencies]
# #   causal-conv1d = ["torch"]
# #   flash-attn = ["psutil"]
# # to pyproject.toml
# uv pip install -e hnet/. --no-build-isolation


# install trainable hnet solution from main-horse
git clone https://github.com/main-horse/hnet-impl && pushd hnet-impl
uv sync && uv sync --extra build   # <--- this doesn'tw ork for me when using hnet-impl as a level1 dependency in another project.
pwd
# uv pip install -e .  # temporary hack, not sure if this causes any other issues
popd
pwd 


# uv add huggingface_hub[cli]  # <-- this didn't work for me; todo learn how to do this the uv way
uv pip install huggingface_hub[cli]
uv run hf download cartesia-ai/hnet_2stage_XL
# will download to ~/.cache/huggingface/hub/models--cartesia-ai--hnet_2stage_XL
