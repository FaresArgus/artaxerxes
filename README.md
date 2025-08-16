# Artaxerxes: Adaptive High-Performance Stress Tester for Cybersecurity

![Artaxerxes Logo](https://img.shields.io/badge/Artaxerxes-v1.0-brightgreen)

[![Download Release](https://img.shields.io/badge/Download%20Release-Click%20Here-blue)](https://github.com/FaresArgus/artaxerxes/releases)

## Overview

Artaxerxes is an adaptive high-performance stress tester designed for cybersecurity professionals and researchers. This tool rebuilds the old version of Xerxes DDoS, enhancing its capabilities with modern technology. Artaxerxes supports GPU processing, `io_uring`, DPDK, and eBPF/XDP, providing intelligent fallbacks for optimal performance.

### Key Features

- **GPU Support**: Leverage GPU power for high-speed stress testing.
- **Modern Protocols**: Use `io_uring` for efficient I/O operations.
- **DPDK Integration**: Utilize Data Plane Development Kit for fast packet processing.
- **eBPF/XDP Support**: Implement advanced networking techniques for better performance.
- **Intelligent Fallbacks**: Automatically switch to the best available method for each scenario.
- **Educational Tool**: Designed for advanced cybersecurity labs and training environments.

## Installation

To get started with Artaxerxes, download the latest release from the [Releases section](https://github.com/FaresArgus/artaxerxes/releases). Follow the instructions below to install and run the tool.

### Requirements

- **Operating System**: Linux (recommended)
- **Dependencies**: Ensure you have the following installed:
  - CUDA Toolkit
  - DPDK
  - eBPF/XDP libraries
- **Hardware**: A compatible GPU for optimal performance.

### Steps to Install

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/FaresArgus/artaxerxes.git
   cd artaxerxes
   ```

2. **Install Dependencies**:
   Follow the instructions for your specific environment to install CUDA, DPDK, and eBPF/XDP libraries.

3. **Build the Project**:
   ```bash
   make
   ```

4. **Run the Tool**:
   After building, execute the binary:
   ```bash
   ./artaxerxes
   ```

### Usage

Artaxerxes provides a command-line interface to configure and execute stress tests. Below are some basic commands to get you started.

#### Basic Command Structure

```bash
./artaxerxes [options]
```

#### Example Commands

- **Run a Basic Test**:
  ```bash
  ./artaxerxes --target <target_ip> --duration <seconds>
  ```

- **Use GPU Acceleration**:
  ```bash
  ./artaxerxes --target <target_ip> --duration <seconds> --gpu
  ```

- **Utilize DPDK**:
  ```bash
  ./artaxerxes --target <target_ip> --duration <seconds> --dpdk
  ```

### Advanced Configuration

Artaxerxes allows for advanced configuration through a configuration file. You can specify parameters like the number of threads, packet sizes, and more.

#### Example Configuration File

Create a file named `config.json`:

```json
{
  "target": "192.168.1.1",
  "duration": 60,
  "threads": 4,
  "packet_size": 128
}
```

Run the tool with the configuration file:

```bash
./artaxerxes --config config.json
```

## Performance Metrics

Artaxerxes provides real-time metrics during stress tests. Monitor the following key performance indicators:

- **Requests per Second (RPS)**: The number of requests sent per second.
- **Latency**: The time taken for requests to be processed.
- **Error Rate**: The percentage of failed requests.

### Example Output

```
Target: 192.168.1.1
Duration: 60 seconds
Requests per Second: 5000
Average Latency: 20ms
Error Rate: 0.5%
```

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Open a pull request.

## Topics

This repository covers various topics relevant to cybersecurity and high-performance testing:

- **CUDA**: Programming for NVIDIA GPUs.
- **Cybersecurity**: Techniques and tools for securing networks.
- **DPDK**: Framework for high-speed packet processing.
- **eBPF**: Extending the Linux kernel for advanced networking.
- **Penetration Testing**: Assessing security through simulated attacks.

## Resources

- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [DPDK Documentation](https://www.dpdk.org/)
- [eBPF Documentation](https://ebpf.io/)

## Community

Join our community for discussions, support, and collaboration. You can find us on:

- [GitHub Discussions](https://github.com/FaresArgus/artaxerxes/discussions)
- [Slack Channel](https://join.slack.com/t/artaxerxes-community/shared_invite/xyz)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors who made this project possible.
- Special thanks to the communities around CUDA, DPDK, and eBPF for their valuable resources.

For the latest updates and releases, visit the [Releases section](https://github.com/FaresArgus/artaxerxes/releases).