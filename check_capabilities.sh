#!/bin/bash
echo "=== Artaxerxes Capability Check ==="

# CUDA
if nvidia-smi > /dev/null 2>&1; then
    echo "[✓] CUDA: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits) GPUs"
else
    echo "[✗] CUDA: Not available"
fi

# DPDK compatible NICs
if lspci | grep -E "(Intel.*X[0-9]|Mellanox|Broadcom)" > /dev/null; then
    echo "[✓] DPDK: Compatible NIC detected"
else
    echo "[✗] DPDK: No compatible NIC"
fi

# io_uring support
if [ -e /sys/kernel/debug/tracing/events/io_uring ]; then
    echo "[✓] io_uring: Kernel support available"
else
    echo "[?] io_uring: Check kernel version (need 5.1+)"
fi

# Root privileges for XDP
if [ $EUID -eq 0 ]; then
    echo "[✓] XDP/eBPF: Root privileges available"
else
    echo "[✗] XDP/eBPF: Need root or capabilities"
fi

echo "=== System Resources ==="
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/{print $2}')"
echo "Network interfaces: $(ip link | grep -c '^[0-9]')"
EOF

chmod +x check_capabilities.sh
./check_capabilities.sh
