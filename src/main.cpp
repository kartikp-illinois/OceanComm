#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <csignal>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include "fir_filter.h"

constexpr int PORT = 8080;
constexpr int BUFFER_SIZE = 4096;
constexpr int SOCKET_TIMEOUT_SEC = 5;
constexpr int MAX_PACKET_FLOATS = BUFFER_SIZE / sizeof(float);

volatile sig_atomic_t shutdown_requested = 0;

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        shutdown_requested = 1;
        std::cout << "\nShutdown signal received. Closing server..." << std::endl;
    }
}

struct ClientState {
    FIRFilter filter;
};

int main() {
    std::cout << "FIR filter UDP server starting on port " << PORT << std::endl;

    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Create UDP socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // Allow immediate reuse of address
    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    // Set receive timeout
    timeval tv{SOCKET_TIMEOUT_SEC, 0};
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Bind socket
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if (bind(sockfd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        std::cerr << "Error binding socket: " << std::strerror(errno) << std::endl;
        close(sockfd);
        return 1;
    }

    std::cout << "Server listening on UDP port " << PORT << std::endl;

    std::vector<float> input(MAX_PACKET_FLOATS);
    std::vector<float> output(MAX_PACKET_FLOATS);
    std::map<std::string, ClientState> clients;

    while (!shutdown_requested) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        ssize_t bytes_received = recvfrom(sockfd, reinterpret_cast<char*>(input.data()), BUFFER_SIZE,
                    0, reinterpret_cast<sockaddr*>(&client_addr), &client_len);

        if (bytes_received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue; // timeout, loop again
            }
            std::cerr << "Receive error: " << std::strerror(errno) << std::endl;
            continue;
        }

        // Validate packet size
        if (bytes_received % sizeof(float) != 0) {
            std::cerr << "Invalid packet size" << std::endl;
            continue;
        }

        int num_samples = bytes_received / sizeof(float);
        std::string client_key =
        std::string(inet_ntoa(client_addr.sin_addr)) + ":" + std::to_string(ntohs(client_addr.sin_port));

        // Get or create filter for client
        FIRFilter& filter = clients[client_key].filter;

        for (int i = 0; i < num_samples; ++i) {
            output[i] = filter.process(input[i]);
        }

        // Send filtered data back
        ssize_t bytes_sent = sendto(sockfd, reinterpret_cast<char*>(output.data()), 
                bytes_received, 0, reinterpret_cast<sockaddr*>(&client_addr), client_len);

        if (bytes_sent < 0) {
            std::cerr << "Send error: " << std::strerror(errno) << std::endl;
        }
    }

    close(sockfd);
    std::cout << "Server shutdown complete." << std::endl;
    return 0;
}
