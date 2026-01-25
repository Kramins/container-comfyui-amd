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


group "default" {
  targets = ["comfyui-amd"]
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