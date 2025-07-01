#!/bin/bash

echo "🚀 Artaxerxes Quick Deploy"
echo "==============================="

# Check system
echo "1. Checking system requirements..."

# Check CUDA
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "   ✅ NVIDIA CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "   ❌ NVIDIA CUDA not found"
fi

# Check kernel version for io_uring
KERNEL_VER=$(uname -r | cut -d. -f1-2)
if [ "$(echo "$KERNEL_VER >= 5.1" | bc 2>/dev/null)" = "1" ]; then
    echo "   ✅ Kernel $KERNEL_VER supports io_uring"
else
    echo "   ⚠️  Kernel $KERNEL_VER may not support io_uring (need 5.1+)"
fi

# Check root privileges
if [ "$EUID" -eq 0 ]; then
    echo "   ✅ Root privileges available for XDP/eBPF"
else
    echo "   ⚠️  Not running as root, XDP/eBPF will be disabled"
fi

echo
echo "2. Installing dependencies..."
apt-get update
apt-get install -y build-essential liburing-dev libbpf-dev pkg-config

echo
echo "3. Building Artaxerxes..."
make

echo
echo "4. Ready to launch!"
echo "   Basic usage: ./Artaxerxes <target> <port>"
echo "   Advanced:    ./Artaxerxes <target> <port> <traffic_spec>"
echo
echo "🎯 Example commands:"
echo "   ./Artaxerxes 192.168.1.100 80"
echo "   ./Artaxerxes 192.168.1.100 80 10M_pps"
echo "   ./Artaxerxes 192.168.1.100 80 5Gbps"
