variable "COMFYUI_VERSIONS" {
  default = "git"
}

variable "DOCKER_REPO" {
  default = "ghcr.io/kramins"
}

variable "ADD_LATEST_TAG" {
  default = "false"
}

variable "USER_ID" {
  default = "1000"
}

variable "GROUP_ID" {
  default = "1000"
}

# vosen/ZLUDA v6-preview — compiled against ROCm 6.4.x HIP ABI, matching rocm/dev-ubuntu-24.04:latest (ROCm 6.4.1).
# ZLUDA devcontainer targets ROCm 6.4.4, so v6-preview binaries are ABI-compatible with our base image.
# The ZLUDA project recommends using the latest pre-release over the v5 stable.
variable "ZLUDA_REPO" {
  default = "vosen/ZLUDA"
}

# v6-preview.59 release tag (latest pre-release as of 2026-03-02).
variable "ZLUDA_VERSION" {
  default = "v6-preview.59"
}

# Archive filename — must stay in sync with ZLUDA_VERSION.
# v6-preview.59 -> zluda-linux-bd82694.tar.gz
variable "ZLUDA_ARCHIVE" {
  default = "zluda-linux-bd82694.tar.gz"
}


group "default" {
  targets = ["comfyui-amd"]
}

group "zluda" {
  targets = ["comfyui-amd-zluda"]
}

target "comfyui-amd" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  name = "comfyui-amd-${replace(version, ".", "-" )}"
  matrix = {
    version = split(",", COMFYUI_VERSIONS)
  }
  args = {
    comfyui_version = "${version}"
    user_id = "${USER_ID}"
    group_id = "${GROUP_ID}"
  }
  output = ["type=docker"]
  no-cache = false
  tags = concat(
    [
      "${DOCKER_REPO}/comfyui-amd:${version}",
    ],
    # Add 'latest' tag if ADD_LATEST_TAG is true and this is not the git version
    ADD_LATEST_TAG == "true" && version != "git" ? ["${DOCKER_REPO}/comfyui-amd:latest"] : []
  )
  labels = {
    "org.opencontainers.image.title" = "ComfyUI AMD"
    "org.opencontainers.image.description" = "ComfyUI with AMD ROCm support"
    "org.opencontainers.image.version" = "${version}"
    "org.opencontainers.image.source" = "https://github.com/kramins/container-comfyui-amd"
    "org.opencontainers.image.licenses" = "MIT"
    "comfyui.version" = "${version}"
  }
}

target "comfyui-amd-zluda" {
  context = "."
  dockerfile = "Dockerfile.zluda"
  platforms = ["linux/amd64"]
  name = "comfyui-amd-zluda-${replace(version, ".", "-")}"
  matrix = {
    version = split(",", COMFYUI_VERSIONS)
  }
  args = {
    comfyui_version = "${version}"
    user_id = "${USER_ID}"
    group_id = "${GROUP_ID}"
    ZLUDA_REPO    = ZLUDA_REPO
    ZLUDA_VERSION = ZLUDA_VERSION
    ZLUDA_ARCHIVE = ZLUDA_ARCHIVE
  }
  output = ["type=docker"]
  no-cache = false
  tags = concat(
    [
      "${DOCKER_REPO}/comfyui-amd-zluda:${version}",
    ],
    ADD_LATEST_TAG == "true" && version != "git" ? ["${DOCKER_REPO}/comfyui-amd-zluda:latest"] : []
  )
  labels = {
    "org.opencontainers.image.title" = "ComfyUI AMD ZLUDA"
    "org.opencontainers.image.description" = "ComfyUI with AMD GPU support via ZLUDA (CUDA on HIP)"
    "org.opencontainers.image.version" = "${version}"
    "org.opencontainers.image.source" = "https://github.com/kramins/container-comfyui-amd"
    "org.opencontainers.image.licenses" = "MIT"
    "comfyui.version" = "${version}"
  }
}