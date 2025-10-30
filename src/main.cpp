/*
 * How to send packets from another computer or network to this FIR filter UDP server:
 * 
 * 1. This server listens on all network interfaces of the machine, 
 *    so it can receive UDP packets sent to any of its IP addresses.
 * 
 * 2. Remote computers must send UDP packets to the actual network IP address of this server,
 *    e.g., 192.168.1.x for local network or the public IP for Internet.
 *    The server does NOT accept packets sent to localhost (127.0.0.1) from other devices.
 * 
 * 3. Make sure that firewalls or routers allow UDP packets on port 8080:
 *    - The server's machine firewall must permit inbound UDP traffic on port 8080.
 *    - If behind a router/NAT, configure port forwarding to tunnel UDP port 8080 to the server IP.
 * 
 * 4. The client on another computer must:
 *    - Send UDP packets to this server's IP address and port 8080.
 *    - Pack float sample data appropriately into the packet payload.
 * 
 * 5. UDP automatically includes sender's IP and port metadata.
 *    This server keeps track of each client by their IP:port to maintain state.
 * 
 * 6. The server processes incoming float data for each client independently,
 *    applies the FIR filter, then sends the filtered results back via UDP to the sender.
 * 
 * 7. No additional message metadata or connection setup is needed from the client.
 * 
 * Summary:
 * - Use this server IP and port in the client.
 * - Ensure network/firewalls allow UDP port 8080 traffic.
 * - Send raw float data packed as bytes.
 * - The server processes and replies to each client separately using their IP:port.
 */


#include <iostream>      // For std::cout, std::cerr (console output)
#include <vector>        // For std::vector (dynamic arrays)
#include <map>           // For std::map (key-value storage for client filters)
#include <string>        // For std::string (text manipulation)

#include <csignal>       // For signal handling (Ctrl+C, kill commands)                          POSIX
#include <cstring>       // For std::strerror (convert errno to text)                            POSIX
#include <cerrno>        // For errno (error codes from system calls)                            POSIX
#include <unistd.h>      // For close() (closing file descriptors)                               POSIX
#include <arpa/inet.h>   // For inet_ntoa, htons, ntohs (network byte order conversions)         POSIX
#include <netinet/in.h>  // For sockaddr_in (IPv4 socket address structure)                      POSIX
#include "fir_filter.h"  // Our FIR filter implementation

// ========== CONSTANTS ==========

constexpr int PORT = 8080;  // UDP port number server listens on (0-65535 range)

constexpr int BUFFER_SIZE = 4096;  // Max bytes per UDP packet (1024 floats × 4 bytes/float)
                                    // Chosen to balance efficiency vs latency
                                    // Larger packets = more efficient, but more delay
                                    // Smaller packets = less delay, but more overhead
                                    // 4096 bytes = 32 ms of audio at 128 kbps
                                    // Common MTU (Maximum Transmission Unit) is 1500 bytes, but UDP can handle larger packets 
                                             // without fragmentation on most modern networks
                                    // Keep under typical MTU to avoid fragmentation issues
                                    // 4096 bytes is safe for most networks


constexpr int SOCKET_TIMEOUT_SEC = 5;  // recvfrom() times out after 5 seconds
                                        // Allows checking shutdown_requested periodically

constexpr int MAX_PACKET_FLOATS = BUFFER_SIZE / sizeof(float);  // = 1024 floats max per packet
                                                                  // sizeof(float) = 4 bytes

// ========== GLOBAL STATE ==========

// Signal handler needs global access to this variable
// volatile: tells compiler value can change unexpectedly (by signal handler)
// sig_atomic_t: guarantees atomic read/write (no partial updates)
volatile sig_atomic_t shutdown_requested = 0;  // 0 = keep running, 1 = shutdown

// ========== SIGNAL HANDLER ==========

// Called asynchronously when user presses Ctrl+C or sends kill signal
// sig: signal number (SIGINT=2 for Ctrl+C, SIGTERM=15 for kill)
void signal_handler(int sig) {
    // Check if it's an interrupt or termination signal
    if (sig == SIGINT || sig == SIGTERM) {
        shutdown_requested = 1;  // Set flag to exit main loop gracefully
        std::cout << "\nShutdown signal received. Closing server..." << std::endl;
    }
    // Note: Can't do complex operations here (signal handler restrictions)
    // Just set flag and return quickly
}

// ========== CLIENT STATE ==========

// Wrapper struct to hold per-client data
// Each connected client gets their own instance
struct ClientState {
    FIRFilter filter;  // Each client has independent filter with own delay_line
                       // This maintains continuity across UDP packets
};

// ========== MAIN FUNCTION ==========

int main() {
    std::cout << "FIR filter UDP server starting on port " << PORT << std::endl;

    // ===== STEP 1: REGISTER SIGNAL HANDLERS =====
    
    // std::signal: register function to call when signal arrives
    // SIGINT: Interrupt signal (Ctrl+C in terminal)
    std::signal(SIGINT, signal_handler);
    
    // SIGTERM: Termination signal (from 'kill' command)
    std::signal(SIGTERM, signal_handler);
    
    // Now when user presses Ctrl+C, signal_handler() is called
    // which sets shutdown_requested = 1

    // ===== STEP 2: CREATE UDP SOCKET =====
    
    // socket(): system call to create network endpoint
    // Returns: file descriptor (integer >= 0) or -1 on error
    // AF_INET: Address Family = IPv4 (vs AF_INET6 for IPv6)
    // SOCK_DGRAM: Socket Type = UDP (vs SOCK_STREAM for TCP)
    // 0: Protocol = let kernel choose (UDP in this case)
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    
    // Check if socket creation failed
    if (sockfd < 0) {
        // std::strerror(errno): converts error number to human-readable string
        // errno: global variable set by failed system calls
        std::cerr << "Error creating socket: " << std::strerror(errno) << std::endl;
        return 1;  // Exit with error code
    }
    // Now sockfd is our "handle" to the socket (like a file descriptor)
    // Think of it as "Window #3" at the post office

    // ===== STEP 3: SET SOCKET OPTIONS =====
    
    // Option 1: SO_REUSEADDR - Allow immediate reuse of port
    int reuse = 1;  // 1 = enable, 0 = disable
    
    // setsockopt(): configure socket behavior
    // sockfd: which socket to configure
    // SOL_SOCKET: level = generic socket options (not protocol-specific)
    // SO_REUSEADDR: option name = allow address reuse
    // &reuse: pointer to value (1 = enable)
    // sizeof(reuse): size of value (4 bytes for int)
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    // Why? After server stops, port stays in TIME_WAIT (~60 sec)
    // Without this, restart would fail with "Address already in use"
    // With this, can restart immediately during development

    // Option 2: SO_RCVTIMEO - Set receive timeout
    // timeval: structure for time (seconds + microseconds)
    timeval tv{SOCKET_TIMEOUT_SEC, 0};  // 5 seconds, 0 microseconds
    
    // Set timeout on receive operations
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    // Why? Without timeout, recvfrom() blocks forever
    // Can't check shutdown_requested flag if blocked!
    // With timeout, recvfrom() returns every 5 sec so we can check flag

    // ===== STEP 4: BIND SOCKET TO ADDRESS =====
    
    // sockaddr_in: IPv4 socket address structure
    // Contains: address family, IP address, port number
    sockaddr_in server_addr{};  // {} zero-initializes all fields
    
    // Set address family to IPv4
    server_addr.sin_family = AF_INET;
    
    // Set IP address to INADDR_ANY (0.0.0.0)
    // Means: listen on ALL network interfaces (WiFi, Ethernet, loopback, etc.)
    // Alternative: inet_addr("127.0.0.1") to listen only on localhost
    server_addr.sin_addr.s_addr = INADDR_ANY;
    
    // Set port number
    // htons(): Host TO Network Short (16-bit)
    // Converts port from host byte order (little-endian on x86) 
    // to network byte order (big-endian standard)
    // Example: 8080 = 0x1F90 → htons → 0x901F (bytes swapped)
    server_addr.sin_port = htons(PORT);
    
    // bind(): associate socket with specific address/port
    // sockfd: which socket to bind
    // (sockaddr*): generic socket address pointer (cast from sockaddr_in*)
    // sizeof(server_addr): size of address structure
    if (bind(sockfd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        std::cerr << "Error binding socket: " << std::strerror(errno) << std::endl;
        close(sockfd);  // Clean up: close socket before exiting
        return 1;
    }
    
    // Success! Kernel now routes packets arriving at UDP:8080 to this socket
    std::cout << "Server listening on UDP port " << PORT << std::endl;

    // ===== STEP 5: ALLOCATE BUFFERS =====
    
    // Input buffer: stores received samples (up to 1024 floats)
    // std::vector automatically manages memory (heap allocation)
    std::vector<float> input(MAX_PACKET_FLOATS);   // Reserve 1024 floats
    
    // Output buffer: stores filtered samples (same size as input)
    std::vector<float> output(MAX_PACKET_FLOATS);  // Reserve 1024 floats
    
    // These are reused for every packet (no reallocation per packet)

    // Per-client state storage
    // std::map: key-value container (like a dictionary)
    // Key: string = "IP:PORT" (e.g., "127.0.0.1:54321")
    // Value: ClientState struct containing FIRFilter
    std::map<std::string, ClientState> clients;
    
    // Why? Each client needs independent filter state
    // Client A's delay_line must not mix with Client B's
    // Map grows automatically as new clients connect

    // ===== STEP 6: MAIN SERVER LOOP =====
    
    // Loop until shutdown_requested becomes 1 (from signal handler)
    while (!shutdown_requested) {
        
        // ===== STEP 6A: PREPARE TO RECEIVE =====
        
        // Structure to store sender's address (filled by recvfrom)
        sockaddr_in client_addr{};  // Zero-initialize
        
        // Size of address structure (input/output parameter)
        // Input: tells recvfrom max size available
        // Output: recvfrom updates this with actual size used
        socklen_t client_len = sizeof(client_addr);

        // ===== STEP 6B: RECEIVE UDP PACKET (BLOCKING CALL) =====
        
        // recvfrom(): receive data from UDP socket
        // This BLOCKS (sleeps) until:
        //   1. Packet arrives, OR
        //   2. Timeout (5 sec) expires
        // 
        // Parameters:
        //   sockfd: which socket to receive from
        //   input.data(): buffer to store received bytes
        //   BUFFER_SIZE: maximum bytes to receive (4096)
        //   0: flags (none used here)
        //   &client_addr: OUTPUT - kernel fills with sender's IP:port
        //   &client_len: INPUT/OUTPUT - size of address structure
        // 
        // Returns: number of bytes received, or -1 on error
        ssize_t bytes_received = recvfrom(
            sockfd,                                      // Our UDP socket
            reinterpret_cast<char*>(input.data()),       // Where to put data (cast float* to char*) //why cast? because recvfrom expects a char* buffer
            BUFFER_SIZE,                                 // Max bytes (4096)
            0,                                           // No special flags
            reinterpret_cast<sockaddr*>(&client_addr),  // Gets filled with sender address
            &client_len                                  // Size of client_addr structure
        );
        
        // After this line (if successful):
        //   - bytes_received = actual bytes received (e.g., 400 for 100 floats)
        //   - input[] contains the data
        //   - client_addr.sin_addr = sender's IP address
        //   - client_addr.sin_port = sender's port number

        // ===== STEP 6C: ERROR HANDLING =====
        
        // Check if recvfrom failed
        if (bytes_received < 0) {
            // errno: global error code set by failed system call
            
            // EAGAIN or EWOULDBLOCK: timeout expired (not a real error)
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;  // Loop back to check shutdown_requested
            }
            
            // Real error occurred (network issue, etc.)
            std::cerr << "Receive error: " << std::strerror(errno) << std::endl;
            continue;  // Skip this iteration, try again
        }

        // ===== STEP 6D: VALIDATE PACKET =====
        
        // Check if bytes received is multiple of sizeof(float)
        // Valid: 400 bytes = 100 floats (400 % 4 = 0) ✓
        // Invalid: 399 bytes = 99.75 floats (399 % 4 = 3) ✗
        if (bytes_received % sizeof(float) != 0) {
            std::cerr << "Invalid packet size" << std::endl;
            continue;  // Ignore malformed packet
        }
        
        // Calculate number of samples in packet
        // Example: 400 bytes / 4 bytes per float = 100 samples
        int num_samples = bytes_received / sizeof(float);

        // ===== STEP 6E: IDENTIFY CLIENT =====
        
        // Create unique key for this client: "IP:port"
        // inet_ntoa(): converts binary IP (in_addr) to dotted string "127.0.0.1"
        // ntohs(): Network TO Host Short - converts port from network to host byte order
        std::string client_key =
            std::string(inet_ntoa(client_addr.sin_addr)) +  // IP as string
            ":" +                                            // Separator
            std::to_string(ntohs(client_addr.sin_port));   // Port as string
        
        // Example result: "127.0.0.1:54321"

        // ===== STEP 6F: GET OR CREATE CLIENT'S FILTER =====
        
        // Access map with client_key
        // If key exists: returns reference to existing ClientState
        // If key doesn't exist: creates new ClientState (calls FIRFilter constructor)
        //   - Allocates delay_line[32] = all zeros
        //   - Calculates filter coefficients
        //   - Sets write_index = 0
        // 
        // .filter: access FIRFilter member of ClientState struct
        FIRFilter& filter = clients[client_key].filter;
        
        // This reference points to the client's personal filter
        // Stays in memory between packets (maintains continuity!)

        // ===== STEP 6G: PROCESS EACH SAMPLE =====
        
        // Loop through all samples in this packet
        for (int i = 0; i < num_samples; ++i) {
            // filter.process(): 
            //   1. Write input[i] to circular buffer at write_index
            //   2. Compute convolution using symmetric optimization
            //   3. Increment write_index (with wraparound)
            //   4. Return filtered value
            output[i] = filter.process(input[i]);
            //when write_index is initally empty, the first 32 samples will be processed with zeros in the delay line
            //the delay line's purpose is to store the most recent 32 samples for the FIR filter
        }
        
        // After loop:
        //   - output[0..num_samples-1] contains filtered results
        //   - filter's delay_line contains last 32 samples
        //   - filter's write_index advanced by num_samples (mod 32)

        // ===== STEP 6H: SEND RESPONSE BACK =====
        
        // sendto(): send data via UDP socket
        // Parameters:
        //   sockfd: which socket to send from
        //   output.data(): data to send (cast float* to char*)
        //   bytes_received: send same number of bytes we received
        //   0: flags (none) --> some possible flags are
        //     - MSG_CONFIRM: only for UDP, allows to send to a broadcast address
        //       (a broadcast address is an address that receives packets sent to all hosts on a network)
        //     - MSG_DONTROUTE: don't use routing, send directly to interface
        //   &client_addr: destination address (SAME as sender!)
        //   client_len: size of destination address structure
        // 
        // Returns: bytes sent, or -1 on error
        ssize_t bytes_sent = sendto(
            sockfd,                                      // Same socket we received on
            reinterpret_cast<char*>(output.data()),     // Filtered data (cast to char*)
            bytes_received,                              // Send same size back (400 bytes)
            0,                                           // No special flags
            reinterpret_cast<sockaddr*>(&client_addr),  // Send TO whoever sent TO us
            client_len                                   // Size of address
        );
        //how is data sent back to the client?
        // The sendto() function is used to send the filtered data back to the client.
        // It takes the following parameters:
        //   - sockfd: the socket file descriptor to send data from
        //   - output.data(): a pointer to the data to send (cast to char*)
        //   - bytes_received: the number of bytes to send
        //   - 0: flags (none) 
            // Some possible flags are
            //   - MSG_CONFIRM: only for UDP, allows to send to a broadcast address
            //     (a broadcast address is an address that receives packets sent to all hosts on a network)
            //   - MSG_DONTROUTE: don't use routing, send directly to interface
                    // routing is what happens when a packet is sent from one network to another network
        //   - &client_addr: the destination address (same as sender)
        //   - client_len: the size of the destination address structure
        //how is data compressed before sending?
        // In this implementation, the data is not compressed before sending.
        // The server sim`ply processes the received samples using the FIR filter
        // and sends the filtered samples back to the client in the same format.
        //what is the specific way the data is sent? the server does something to receive to grab the metadata or smth?
        // The data is sent back to the client using the sendto() function,
        // which sends the filtered samples to the address stored in client_addr.
        // The client_addr structure contains the IP address and port number of the client,
        // allowing the server to send the response directly back to the sender.
        //unrolling the loop for processing samples?
        // In this implementation, the loop for processing samples is not unrolled.
        // The server processes each sample one by one in a simple for loop.
        // Loop unrolling could be considered for optimization in performance-critical applications,
        // but it is not implemented here for simplicity and clarity.


        // Check if send failed
        if (bytes_sent < 0) {
            std::cerr << "Send error: " << std::strerror(errno) << std::endl;
            // Don't exit - just note error and continue serving other clients
        }
        
        // Success! Client should receive filtered data now
        // Typical round-trip: ~2ms on localhost
        
    }  // End of while loop - goes back to recvfrom() for next packet

    // ===== STEP 7: CLEANUP (after shutdown_requested = 1) =====
    
    // close(): release socket file descriptor
    // Tells kernel we're done with this socket
    // Port 8080 becomes available again
    close(sockfd);
    
    std::cout << "Server shutdown complete." << std::endl;
    
    // Destructors automatically called:
    //   - std::map<> destroys all ClientState objects
    //   - Each ClientState destructor destroys its FIRFilter
    //   - Each FIRFilter destructor deallocates delay_line vector
    //   - input and output vectors deallocated
    
    return 0;  // Success!
}