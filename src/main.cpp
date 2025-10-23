#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <map>
#include "fir_filter.h"

constexpr int PORT = 8080;
constexpr int BUFFER_SIZE = 4096;
constexpr int MAX_PACKET_FLOATS = BUFFER_SIZE / sizeof(float);
constexpr int SOCKET_TIMEOUT_SEC = 5;

// Global flag for graceful shutdown
volatile sig_atomic_t shutdown_requested = 0;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        shutdown_requested = 1;
        std::cout << "\nShutdown signal received. Closing server..." << std::endl;
    }
}

// Client state manager for multi-client support with independent filters
class ClientState {
public:
    FIRFilter filter;
    struct sockaddr_in addr;
    socklen_t addr_len;
    
    ClientState() : addr_len(sizeof(addr)) {
        std::memset(&addr, 0, addr_len);
    }
};

int main() {
    std::cout << "FIR filter UDP application starting on port " << PORT << std::endl;

    // Register signal handlers for graceful shutdown
    if (std::signal(SIGINT, signal_handler) == SIG_ERR ||
        std::signal(SIGTERM, signal_handler) == SIG_ERR) {
        std::cerr << "Warning: Failed to register signal handlers" << std::endl;
    }

    // Create UDP socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket: " << std::strerror(errno) << std::endl;
        return -1;
    }

    // Enable socket address reuse to avoid "Address already in use" error
    int reuse_addr = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr)) < 0) {
        std::cerr << "Warning: Failed to set SO_REUSEADDR: " << std::strerror(errno) << std::endl;
    }

    // Set socket receive timeout
    struct timeval tv;
    tv.tv_sec = SOCKET_TIMEOUT_SEC;
    tv.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        std::cerr << "Warning: Failed to set receive timeout: " << std::strerror(errno) << std::endl;
    }

    // Configure server address
    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Bind socket
    if (bind(sockfd, (const struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error binding socket to port " << PORT << ": " << std::strerror(errno) << std::endl;
        close(sockfd);
        return -1;
    }

    std::cout << "UDP server listening on port " << PORT << std::endl;

    // Allocate input and output buffers
    char* input_buffer = new char[BUFFER_SIZE];
    char* output_buffer = new char[BUFFER_SIZE];

    if (!input_buffer || !output_buffer) {
        std::cerr << "Error allocating buffers" << std::endl;
        close(sockfd);
        return -1;
    }

    // Map to track per-client filters (supports multiple independent clients)
    std::map<std::string, ClientState> clients;

    while (!shutdown_requested) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        std::memset(&client_addr, 0, client_len);

        // Receive data from client
        ssize_t bytes_received = recvfrom(sockfd, input_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &client_len);

        // Handle timeout or no data received
        if (bytes_received == 0) {
            continue; // timeout occurred
        }

        if (bytes_received < 0) {
            // EAGAIN/EWOULDBLOCK on non-blocking or timeout (expected behavior)
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            std::cerr << "Error receiving data: " << std::strerror(errno) << std::endl;
            continue;
        }

        // Validate packet size: must be multiple of sizeof(float)
        if (bytes_received % sizeof(float) != 0) {
            std::cerr << "Warning: Received packet of size " << bytes_received 
                      << " is not a multiple of sizeof(float) = " << sizeof(float) << std::endl;
            continue; // Skip malformed packet
        }

        int num_samples = bytes_received / sizeof(float);

        // Safety check: ensure buffer is large enough
        if (num_samples > MAX_PACKET_FLOATS) {
            std::cerr << "Error: Packet contains " << num_samples 
                      << " samples, exceeds maximum of " << MAX_PACKET_FLOATS << std::endl;
            continue;
        }

        // Get or create client state (per-client filter for multi-client support)
        std::string client_key = std::string(inet_ntoa(client_addr.sin_addr)) + ":" 
                                 + std::to_string(ntohs(client_addr.sin_port));
        
        if (clients.find(client_key) == clients.end()) {
            std::cout << "New client connected: " << client_key << std::endl;
            clients[client_key].addr = client_addr;
            clients[client_key].addr_len = client_len;
        }

        ClientState& client_state = clients[client_key];
        FIRFilter& filter = client_state.filter;

        // Cast buffers to float pointers
        float* input_samples = reinterpret_cast<float*>(input_buffer);
        float* output_samples = reinterpret_cast<float*>(output_buffer);


        // Process each sample through the filter (maintains state across packets)
        for (int i = 0; i < num_samples; i++) {
            // Check for NaN or Inf in input
            if (!std::isfinite(input_samples[i])) {
                std::cerr << "Warning: Non-finite input value at sample " << i 
                          << " from client " << client_key << ": " << input_samples[i] << std::endl;
                output_samples[i] = 0.0f; // Or handle as error
                continue;
            }

            output_samples[i] = filter.process(input_samples[i]);

            // Check for NaN or Inf in output
            if (!std::isfinite(output_samples[i])) {
                std::cerr << "Warning: Non-finite output value at sample " << i 
                          << " from client " << client_key << ": " << output_samples[i] << std::endl;
                output_samples[i] = 0.0f; // Clamp to safe value
            }
        }

        // Send filtered data back to client
        ssize_t bytes_sent = sendto(sockfd, output_buffer, bytes_received, 0,
                                   (const struct sockaddr*)&client_addr, client_len);

        if (bytes_sent < 0) {
            std::cerr << "Error sending data to client " << client_key << ": " 
                      << std::strerror(errno) << std::endl;
            continue;
        }

        if (bytes_sent != bytes_received) {
            std::cerr << "Warning: Sent " << bytes_sent << " bytes but expected " 
                      << bytes_received << " bytes to client " << client_key << std::endl;
        }
    }

    // Cleanup
    std::cout << "Cleaning up resources..." << std::endl;
    delete[] input_buffer;
    delete[] output_buffer;
    close(sockfd);

    std::cout << "Server shutdown complete" << std::endl;
    return 0;
}
