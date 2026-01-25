# ComfyUI Container with AMD ROCm Support

This project provides Docker containers for running ComfyUI with AMD ROCm support. The containers are automatically built and published to GitHub Container Registry (GHCR) with version management.

## 🚀 Quick Start

### Pull and Run

```bash
# Pull the latest stable release
docker pull ghcr.io/kramins/comfyui-amd:latest

# Or pull the git version (always pulls latest ComfyUI main branch at startup)
docker pull ghcr.io/kramins/comfyui-amd:git

# Or pull a specific version
docker pull ghcr.io/kramins/comfyui-amd:0.8.2

# Run the container
docker run --rm --privileged \
  -v /dev:/dev \
  -p 8188:8188 \
  -v ~/comfyui/models:/data/models \
  -v ~/comfyui/custom_nodes:/data/custom_nodes \
  -v ~/comfyui/output:/data/output \
  -v ~/comfyui/user:/data/user \
  --name comfyui \
  ghcr.io/kramins/comfyui-amd:latest
```

Access ComfyUI at `http://localhost:8188`

## 📦 Available Versions

Images are published to: `ghcr.io/kramins/comfyui-amd`

- **`latest`** - Most recent stable ComfyUI release
- **`git`** - Pulls latest ComfyUI main branch at container startup (always fresh)
- **`X.Y.Z`** - Specific ComfyUI release versions (e.g., `0.8.2`, `0.8.1`)

### Version Behavior

- **Release versions** (`0.8.2`, etc.): ComfyUI is downloaded during image build and remains fixed
- **Git version** (`git`): ComfyUI is cloned/pulled from GitHub main branch **at container startup**, ensuring you always have the latest changes without rebuilding the image

## 🏗️ Building Locally

### Prerequisites

- Docker with buildx support
- AMD ROCm compatible hardware (for running)

### Build Script Usage

```bash
# Build the git version (default)
./build-container.sh

# Build a specific version
./build-container.sh 0.8.2

# Build and push to GHCR (requires authentication)
./build-container.sh git --push
```

### Authenticate to GHCR

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

## 🤖 GitHub Actions Automation

The repository uses GitHub Actions to automatically build new ComfyUI versions:

### Automatic Detection
- On push to main branch, checks for new ComfyUI releases
- Compares against existing GHCR tags
- Builds only new versions automatically
- First run builds the `git` version plus last 5 releases

### Manual Builds
Trigger manual builds via GitHub Actions:
1. Go to Actions → "Build ComfyUI Containers"
2. Click "Run workflow"
3. Optional: Specify version (e.g., `git`, `0.8.2`) or leave empty for auto-detect
4. Optional: Enable "force rebuild" to rebuild existing versions

## 📂 Project Structure

- `build-container.sh`: Local build script with version selection and push support
- `Dockerfile`: Multi-version container definition with AMD ROCm
- `docker-bake.hcl`: Docker bake configuration for matrix builds
- `run-container.sh`: Container run script
- `root/start.sh`: Container startup script (handles git pulls, dependencies, etc.)
- `.github/workflows/build.yml`: Automated build workflow
- `.github/scripts/detect-versions.sh`: Version detection script

## 🔧 Container Configuration

### Volume Mounts

| Mount Point | Description |
|-------------|-------------|
| `/data/models` | ComfyUI model files (checkpoints, VAE, etc.) |
| `/data/custom_nodes` | Custom nodes and extensions |
| `/data/output` | Generated images and outputs |
| `/data/user` | User settings and configurations |
| `/data/input` | Input files for ComfyUI |
| `/data/temp` | Temporary files |

### Docker Run Options

| Option | Description |
|--------|-------------|
| `--rm` | Automatically remove container when it exits |
| `--privileged` | Required for AMD ROCm GPU access |
| `-v /dev:/dev` | Mount devices for GPU access |
| `-p 8188:8188` | Expose ComfyUI web interface |
| `--name comfyui` | Container name for easy management |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_VERSION` | (from build) | ComfyUI version (git or X.Y.Z) |
| `BACKEND` | `rocm` | Compute backend (ROCm for AMD) |
| `ROCM_VERSION` | `6.4` | ROCm version for PyTorch |

## 🔄 Updating

- **Git version**: Restart the container - it automatically pulls latest changes
- **Release versions**: Pull the new image tag from GHCR
- **Local builds**: Run build script with new version

## 📝 License

MIT