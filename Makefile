# Artaxerxes Makefile with conditional compilation
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -flto -pthread -Wall -Wextra
LIBS = -lm -lrt -lpthread

# Feature detection
CUDA_AVAILABLE := $(shell which nvcc >/dev/null 2>&1 && echo 1 || echo 0)
DPDK_AVAILABLE := $(shell pkg-config --exists libdpdk 2>/dev/null && echo 1 || echo 0)
IO_URING_AVAILABLE := $(shell echo '#include <liburing.h>' | gcc -E - >/dev/null 2>&1 && echo 1 || echo 0)
XDP_AVAILABLE := $(shell echo '#include <bpf/libbpf.h>' | gcc -E - >/dev/null 2>&1 && echo 1 || echo 0)

TARGET = artaxerxes
SOURCES = artaxerxes.cu

# Conditional compilation flags
ifeq ($(CUDA_AVAILABLE),1)
    CFLAGS += -DCUDA_AVAILABLE
    LIBS += -lcuda -lcudart -lcurand
    COMPILER = $(NVCC)
    NVCCFLAGS = -O3 -arch=sm_86 --compiler-options="$(CFLAGS)"
else
    COMPILER = $(CC)
    # Rename .cu to .c for CPU-only compilation
    SOURCES = artaxerxes.c
endif

ifeq ($(DPDK_AVAILABLE),1)
    CFLAGS += -DDPDK_AVAILABLE $(shell pkg-config --cflags libdpdk)
    LIBS += $(shell pkg-config --libs libdpdk)
endif

ifeq ($(IO_URING_AVAILABLE),1)
    CFLAGS += -DIO_URING_AVAILABLE
    LIBS += -luring
endif

ifeq ($(XDP_AVAILABLE),1)
    CFLAGS += -DXDP_AVAILABLE
    LIBS += -lbpf -lelf -lz
endif

all: detect $(TARGET)

detect:
	@echo "🔍 Feature Detection:"
	@echo "====================="
	@echo "CUDA:        $(if $(filter 1,$(CUDA_AVAILABLE)),✅ Available,❌ Not available)"
	@echo "DPDK:        $(if $(filter 1,$(DPDK_AVAILABLE)),✅ Available,❌ Not available)"
	@echo "io_uring:    $(if $(filter 1,$(IO_URING_AVAILABLE)),✅ Available,❌ Not available)"
	@echo "XDP/eBPF:    $(if $(filter 1,$(XDP_AVAILABLE)),✅ Available,❌ Not available)"
	@echo ""

# Create C version for CPU-only compilation
artaxerxes.c: artaxerxes.cu
	@if [ "$(CUDA_AVAILABLE)" -eq "0" ]; then \
		echo "🔄 Creating CPU-only version..."; \
		sed 's/__global__/\/\/__global__/g; s/__device__/\/\/__device__/g; s/<<<.*>>>/\/\/&/g' $< > $@; \
	fi

$(TARGET): $(SOURCES)
ifeq ($(CUDA_AVAILABLE),1)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)
else
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)
endif
	@echo "🎯 Compiled with features:"
	@echo "   CUDA: $(CUDA_AVAILABLE), DPDK: $(DPDK_AVAILABLE), io_uring: $(IO_URING_AVAILABLE), XDP: $(XDP_AVAILABLE)"

install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/
	sudo chmod +x /usr/local/bin/$(TARGET)
	@echo "✅ Installed to /usr/local/bin/$(TARGET)"

uninstall:
	sudo rm -f /usr/local/bin/$(TARGET)

clean:
	rm -f $(TARGET) artaxerxes.c

test: $(TARGET)
	@echo "🧪 Running capability test..."
	./$(TARGET) 127.0.0.1 8080 1s

benchmark: $(TARGET)
	@echo "📊 Running performance benchmark..."
	./$(TARGET) 127.0.0.1 8080 10s

docker-build:
	@echo "🐳 Building Docker container with all dependencies..."
	docker build -t artaxerxes .

help:
	@echo "Artaxerxes Build System"
	@echo "============================"
	@echo "Targets:"
	@echo "  all      - Build with auto-detected features"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  test     - Quick functionality test"
	@echo "  bench    - Performance benchmark"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Features automatically detected and enabled:"
	@echo "  CUDA, DPDK, io_uring, XDP/eBPF"

.PHONY: all detect install uninstall clean test benchmark docker-build help
