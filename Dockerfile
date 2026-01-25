FROM rocm/dev-ubuntu-24.04:latest
USER root

ARG user_id=1000
ARG group_id=1000
ARG comfyui_version=git

# Install dependencies
RUN apt update && apt install -y \
    software-properties-common \ 
    wget curl tar \
    python3 python3-pip python3.12-venv \
    git rsync \
    sqlite3 libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user and group
RUN if ! getent group $group_id; then groupadd -g $group_id user; fi && \
    if ! id -u user 2>/dev/null; then \
    if id -u $user_id 2>/dev/null; then \
    usermod -l user -d /home/user -m $(id -nu $user_id); \
    else \
    useradd -m -u $user_id -g $group_id -s /bin/bash user; \
    fi; \
    fi && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -aG video user

USER user

ENV BACKEND=rocm
ENV COMFYUI_VERSION=${comfyui_version}
WORKDIR /app

# Add ComfyUI based off of version, git is a special version
# For git: ComfyUI will be cloned at runtime by start.sh
# For releases: Download and extract the release archive during build
RUN if [ "${comfyui_version}" != "git" ]; then \
    echo "[BUILD] Downloading ComfyUI release v${comfyui_version}..."; \
    curl -L "https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/v${comfyui_version}.tar.gz" -o /tmp/comfyui.tar.gz && \
    tar -xzf /tmp/comfyui.tar.gz -C /app --strip-components=1 && \
    rm /tmp/comfyui.tar.gz && \
    echo "[BUILD] ComfyUI v${comfyui_version} extracted to /app"; \
    fi

# Add the startup script and model packs configuration
ADD root/* /

EXPOSE 8188
SHELL ["/bin/bash", "-c"]
ENV PYTHONUNBUFFERED=1

# Use the startup script to handle all setup and runtime
CMD ["/bin/bash", "/start.sh"]
