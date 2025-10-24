# C++ Design Exercise - Efficient Streaming FIR Filter

## Filter Design  
Assume a discrete filter \( h[n] \) defined as follows:

$$
h[n] =
\begin{cases}
\mathrm{sinc}\left(\frac{B}{F_s} \cdot (n + 0.5)\right) \cdot \left(1 + \cos\left(\pi \cdot \frac{n+0.5}{N+0.5}\right)\right), & \text{when } -N \leq n < N \\
0, & \text{otherwise}
\end{cases}
$$

with

$$
\mathrm{sinc}(x) = \frac{\sin(x)}{x}
$$

$$
F_s = 100 \times 10^{3}
$$

$$
B = 10 \times 10^{3}
$$

$$
N = 16
$$

$$
0 < F_c < 50 \times 10^{3}
$$

$$
x[n] =
\begin{cases}
\sin\left(2 \pi F_c \frac{n}{F_s}\right), & \text{when } n \geq 0 \\
0, & \text{otherwise}
\end{cases}
$$

## Functional Requirements

### The design shall:
1. Have a single UDP socket for receiving and transmitting data  
2. Receive a real-valued (32-bit IEEE float) stream of data via UDP socket as input \( x[n] \), where \( x[n] \) has a sampling rate of \( F_s \). The signal \( x[n] \) is infinitely long and has continuity across UDP packet boundaries  
3. Perform the convolution \( y[n] = x[n] * h[n] \) on-the-fly as samples are received from the UDP socket. The convolution should be continuous across packet boundaries  
4. Send the result \( y[n] \) back on the same UDP socket as a 32-bit IEEE float  

## Implementation Requirements

- Your solution should compile in a Unix environment with POSIX sockets  
- Code must be C++14 compliant  
- Library usage is limited to the C++ Standard Library and the POSIX socket library  
- Provide a high level summary of the resources required by your implementation (number of memory loads/stores, multiplies, additions, etc.)  
- Provide a Python test bench for testing your solution. Designs that fail to pass their own test benches will not be considered. The test bench may use third-party libraries (e.g. NumPy)  

## Exercise Submission  
Submit your design source along with a `README.md` or script that describes the compile steps. Provide a link to a zip file download from Google Drive or similar. Your submission will be compiled and tested for correctness. Please ensure that your solution is submitted within 48 hours of receipt.  

## Next Steps  
The subsequent step of the interview process will allow you to give a short presentation on your solution with follow-up questions and discussion regarding your implementation.
