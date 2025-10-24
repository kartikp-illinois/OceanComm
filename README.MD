# FIR Filter UDP Streaming Server

This solution includes both **normalized** and **unnormalized** coefficient variants.

---

## Overview
This implementation provides a real-time streaming FIR filter that:

* Processes continuous **32-bit IEEE float** audio/signal streams via UDP
* Maintains **filter state across packet boundaries** for seamless streaming
* Uses **symmetric filter optimization** to reduce computational overhead by 50%
* Implements a **circular buffer** for efficient memory usage
* Supports both **normalized (unity DC gain)** and **unnormalized (raw sinc window)** coefficients

### Filter Specification

| Parameter              | Value                                                                  |
| ---------------------- | ---------------------------------------------------------------------- |
| **Filter Length**      | 32 taps (symmetric)                                                    |
| **Bandwidth**          | 10 kHz                                                                 |
| **Sampling Frequency** | 100 kHz                                                                |
| **Window**             | Cosine-tapered sinc window                                             |
| **Architecture**       | Symmetric convolution (16 multiplications vs. 32 without optimization) |

---

**Libraries:**

* C++ Standard Library
* POSIX socket library (included in most Unix systems)

**Python Testing:**

* Python 3.6+
* NumPy
* struct module

---

## Installation

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install cmake g++ python3 python3-pip
pip3 install numpy
```


## 🏗️ Building and Compiling

### Quick Build (Automated)

```bash
chmod +x build.sh
./build.sh
```

### Manual Build

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make

# Executable created at: ./build/fir_filter_udp
```

### Compiler Flags

| Flag            | Purpose                                 |
| --------------- | --------------------------------------- |
| `-O3`           | Aggressive optimization for performance |
| `-Wall -Wextra` | All common warnings enabled             |
| `-std=c++14`    | Requires C++14 standard                 |

---

## ▶️ Running the Application

### Start the UDP Server

```bash
./build/fir_filter_udp
```

**Expected output:**

```
FIR filter UDP server starting on port 8080
```

The server listens on **127.0.0.1:8080** and processes incoming UDP packets.

---

## 📡 Server Protocol

### Input Format

* UDP packets containing 32-bit IEEE float samples
* Sample rate: **100 kHz**
* Packet size: Variable (up to 4096 bytes ≈ 1024 float samples)

### Output Format

* Filtered samples returned as 32-bit IEEE float values
* Same packet structure as input

**Packet Structure:**

```
[float] [float] [float] ... [float]
 sample_1 sample_2 sample_3   sample_n
```

---

## Testing

### Running Tests

Run in a separate terminal while the server is active:

```bash
cd test
python3 test_fir_filter.py
```

### Test Suite Overview

| Test               | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| Impulse Response   | Verifies filter coefficients match specification |
| Direct Convolution | Validates signal processing accuracy             |
| Passband Response  | Confirms passband frequencies pass through       | 
| Stopband Response  | Confirms stopband frequencies are attenuated     |
| Packet Continuity  | Ensures seamless filtering across UDP packets    |

## Resources Used

### Per-Packet Processing

| Operation                  | Count   | Notes                           |
| -------------------------- | ------- | ------------------------------- |
| Memory Loads (per output)  | 48      | 32 delay line + 16 coefficients |
| Memory Stores (per output) | 1       | Write to delay line             |

---

### Computational Requirements

#### Unnormalized Version

```
Operations Per Sample:
  ├─ 16 multiplications
  ├─ 31 additions
  ├─ 48 memory loads
  ├─ 1 memory store
  ├─ 1 bitwise wrap
  ├─ 4 integer ops
≈ 102 operations per output
```

#### Normalized Version

Additional startup cost:

```
32 × sin()
32 × cos()
32 × division
32 × accumulation
≈ 128 ops (one-time)
```

---


## Efficiency

* **Symmetric filter:** 16 multiplications instead of 32 (50% reduction)
* **Memory:** 16 coefficients stored instead of 32
* **Circular buffer optimization:** bitwise wrap (`& FILTER_MASK`) ~10× faster than modulo

---

## Implementation Variants

### Variant 1: Unnormalized Coefficients

**Branch:** `Unnormalized`

* No normalization
* Raw sinc window
* **DC Gain ≠ 0 dB (~31 dB)**
* **Use Case:** Relative filtering (gain unimportant)

**Frequency Response:**

* Passband (0.5–5 kHz): +15 to +30 dB
* Stopband (15 kHz+): −5 to −26 dB

---

### Variant 2: Normalized Coefficients

**Branch:** `Main`

* Coefficients divided by sum for unity DC gain
* **DC Gain = 1.0 (0 dB)**
* **Use Case:** Audio processing, general signal conditioning

**Frequency Response:**

* Passband (0.5–5 kHz): 0 dB ± 3 dB
* Stopband (15 kHz+): −5 to −26 dB

---

---

## UDP Socket Interface

| Parameter       | Value                      |
| --------------- | -------------------------- |
| Protocol        | UDP                        |
| Bind Address    | 127.0.0.1                  |
| Port            | 8080                       |
| Timeout         | 5 seconds                  |
| Max Packet Size | 4096 bytes (~1024 samples) |

### Python Client Example

```python
import socket, struct, numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.connect(('127.0.0.1', 8080))

signal = np.sin(2 * np.pi * 1000 * np.arange(100) / 100e3).astype(np.float32)
data = struct.pack(f'{len(signal)}f', *signal)
sock.send(data)

response = sock.recv(len(data))
output = np.array(struct.unpack(f'{len(signal)}f', response))
sock.close()
```

---
