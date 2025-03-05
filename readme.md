# ComfyUI Container

This project sets up a Docker container for running ComfyUI with AMD ROCm support. The container includes all necessary dependencies and downloads required models and custom nodes.

## Project Structure

- `build-container.sh`: Script to build the Docker image.
- `Dockerfile`: Dockerfile to create the Docker image.
- `run-container.sh`: Script to run the Docker container.

## Setup

### Prerequisites

- Docker installed on your system.
- AMD ROCm compatible hardware.

### Building the Docker Image

To build the Docker image, run:

```bash
./build-container.sh
```

### Running the Docker Container

To run the Docker container, run:


| Option                                      | Description                                                                                     |
|---------------------------------------------|-------------------------------------------------------------------------------------------------|
| `--rm`                                      | Automatically removes the container when it exits.                                              |
| `--privileged`                              | Gives extended privileges to the container.                                                     |
| `-v /dev:/dev`                              | Mounts the host's `/dev` directory to the container's `/dev` directory, allowing device access.  |
| `-p 8188:8188`                              | Maps port 8188 on the host to port 8188 on the container.                                       |
| `-v ~/comfyui/models:/app/models`           | Mounts the host's `~/comfyui/models` directory to the container's `/app/models` directory.      |
| `-v ~/comfyui/custom_nodes:/app/custom_nodes`| Mounts the host's `~/comfyui/custom_nodes` directory to the container's `/app/custom_nodes` directory.|
| `-v ~/comfyui/output:/app/output`           | Mounts the host's `~/comfyui/output` directory to the container's `/app/output` directory.      |
| `-v ~/comfyui/user:/app/user`               | Mounts the host's `~/comfyui/user` directory to the container's `/app/user` directory.          |
| `--name comfyui`                            | Assigns the name `comfyui` to the container.                                                    |
| `kramins/comfyui-amd`                       | Specifies the Docker image to use for the container.                                            |


```bash
./run-container.sh
```