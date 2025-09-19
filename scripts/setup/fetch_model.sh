# uv add huggingface_hub[cli]  # <-- this didn't work for me; todo learn how to do this the uv way
uv pip install huggingface_hub[cli]
uv run hf download cartesia-ai/hnet_2stage_XL
# will download to ~/.cache/huggingface/hub/models--cartesia-ai--hnet_2stage_XL

# if you want to specify the dir:
# uv run hf download cartesia-ai/hnet_2stage_XL --local-dir "models/cartesia-ai/hnet_2stage_XL"


# cp /path/to/hnet/hnet_2stage_XL.pt /path/to/hnet/config/hnet_2stage_XL.json .
cp ~/.cache/huggingface/hub/models--cartesia-ai--hnet_2stage_XL /path/to/hnet/config/hnet_2stage_XL.json .
uv run -m hnet_impl.modeling_hnet