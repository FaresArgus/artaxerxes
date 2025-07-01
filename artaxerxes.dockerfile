FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libnuma-dev \
    libpcap-dev \
    python3-pyelftools \
    libbpf-dev \
    libelf-dev \
    zlib1g-dev \
    liburing-dev \
    net-tools \
    ethtool \
    wget \
    git \
    cmake

# Install DPDK
RUN wget http://fast.dpdk.org/rel/dpdk-22.11.1.tar.xz && \
    tar xf dpdk-22.11.1.tar.xz && \
    cd dpdk-22.11.1 && \
    meson setup build && \
    cd build && \
    ninja && \
    ninja install && \
    ldconfig

# Set environment
ENV PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig

# Copy source
COPY . /app
WORKDIR /app

# Build
RUN make

# Set capabilities for XDP (if running with --privileged)
RUN setcap cap_sys_admin,cap_net_admin+eip artaxerxes || true

ENTRYPOINT ["./artaxerxes"]
