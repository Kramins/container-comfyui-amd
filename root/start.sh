
#!/bin/bash
APP_MODELS_PATH="/app/models"
APP_CUSTOM_NODES_PATH="/app/custom_nodes"
LOAD_MODELS="true"  

# Define Model folder structure
declare -a model_folder_structure=(
    "checkpoints"
    "clip_vision"
    "controlnet"
    "diffusion_models"
    "inpaint"
    "ipadapter"
    "loras"
    "text_encoders"
    "upscale_models"
    "vae"
)

# Create the model folder structure if it doesn't exist
for folder in "${model_folder_structure[@]}"; do
    echo "Checking if folder $APP_MODELS_PATH/$folder exists..."
    if [ ! -d "$APP_MODELS_PATH/$folder" ]; then
        mkdir -p "$APP_MODELS_PATH/$folder"
    fi
done

# Define the mapping of URLs to directories
declare -A url_mapping=(
    # Shared
    ["https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"]="clip_vision/SD1.5/"
    ["https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth"]="upscale_models/"
    ["https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors"]="upscale_models/"
    ["https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X3_DIV2K.safetensors"]="upscale_models/"
    ["https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X4_DIV2K.safetensors"]="upscale_models/"
    ["https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors"]="inpaint/"
    # SD 1.5
    ["https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors"]="controlnet/"
    ["https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"]="controlnet/"
    ["https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"]="ipadapter/"
    ["https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-8steps-CFG-lora.safetensors"]="loras/"

    ["https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors"]="loras/"

    # SD XL
    ["https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"]="ipadapter/"
    ["https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors"]="loras/"
    ["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"]="inpaint/"
    ["https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"]="inpaint/"
    
    # Checkpoints
    ["https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors"]="checkpoints/"
    ["https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors"]="checkpoints/"
    ["https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors"]="checkpoints/"
    ["https://huggingface.co/misri/zavychromaxl_v80/resolve/main/zavychromaxl_v80.safetensors"]="checkpoints/"

)

# Loop through the mapping and create directories and download files if missing


declare -A node_mapping=(
    ["https://github.com/ltdrdata/ComfyUI-Manager"]="comfyui-manager"
)

if [ "$LOAD_MODELS" = "true" ]; then
    for url in "${!url_mapping[@]}"; do
        directory="${url_mapping[$url]}"
        if [ ! -d "$APP_MODELS_PATH/$directory" ]; then
            mkdir -p "$APP_MODELS_PATH/$directory"
        fi
        final_path="$APP_MODELS_PATH/$directory/$(basename "$url")"
        part_path="${final_path}.part"
        if [ ! -f "$final_path" ]; then
            echo "Downloading $(basename "$url") to $part_path"
            if wget -c -O "$part_path" "$url"; then
                mv "$part_path" "$final_path"
                chmod 777 "$final_path"
                echo "Download completed: $(basename "$url")"
            else
                echo "Download failed: $(basename "$url")"
                rm -f "$part_path"  # Clean up partial file on failure
            fi
        else
            echo "Model $(basename "$url") already exists. Skipping download."
        fi
    done
fi




echo "Starting the app"
cd /app/
. ./venv/bin/activate

# Install custom nodes
echo "Installing custom nodes"
for node in "${!node_mapping[@]}"; do
    directory="${node_mapping[$node]}"
    final_path="$APP_CUSTOM_NODES_PATH/$directory"
    if [ ! -d "$final_path" ]; then
        echo "Downloading $(basename "$node") to $final_path"
        git clone $node $final_path
    else
        echo "Node $(basename "$node") already exists. Skipping download."
    fi
done


python3 main.py --listen 0.0.0.0