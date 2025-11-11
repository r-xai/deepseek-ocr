#!/bin/bash

# Test script to find a PyTorch container that supports GB10
# Tries multiple container versions until one works

echo "Testing NVIDIA PyTorch containers for GB10 GPU support..."
echo "==========================================================="
echo ""

# List of containers to try (newest first)
CONTAINERS=(
    "nvcr.io/nvidia/pytorch:25.02-py3"
    "nvcr.io/nvidia/pytorch:25.01-py3"
    "nvcr.io/nvidia/pytorch:24.12-py3"
    "nvcr.io/nvidia/pytorch:24.11-py3"
)

test_container() {
    local image=$1
    echo ""
    echo "Testing: $image"
    echo "-----------------------------------------------------------"

    # Try to run the container with GPU support
    docker run --gpus=all --rm \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        $image python3 - <<'PY' 2>&1
import torch
import sys

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    sys.exit(0)
else:
    print("ERROR: CUDA not available")
    sys.exit(1)
PY

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ SUCCESS! This container works with GB10"
        echo "-----------------------------------------------------------"
        echo ""
        echo "Use this container by editing run_docker.sh:"
        echo "  PYTORCH_IMAGE=\"$image\""
        echo ""
        return 0
    else
        echo "✗ Failed - trying next container..."
        return 1
    fi
}

# Test each container
for container in "${CONTAINERS[@]}"; do
    if test_container "$container"; then
        echo ""
        echo "RECOMMENDATION: Update run_docker.sh with:"
        echo "  PYTORCH_IMAGE=\"$container\""
        exit 0
    fi
done

echo ""
echo "==========================================================="
echo "WARNING: None of the tested containers worked!"
echo ""
echo "Options:"
echo "1. Check Docker GPU support: docker run --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
echo "2. Update NVIDIA Container Toolkit"
echo "3. Check available containers: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch"
echo ""
