#!/usr/bin/env python3

import socket
# socket: Python's network programming module
# Provides access to BSD socket interface (same as C++)
# We'll use this to create UDP sockets and send/receive data

import numpy as np
# numpy (np): Numerical Python library
# Provides fast array operations, mathematical functions
# Essential for signal processing (sin, cos, convolution, etc.)

import struct
# struct: Binary data packing/unpacking
# Converts between Python values and C-style binary data
# We use it to pack floats into bytes for network transmission

import time
# time: Time-related functions
# We use time.perf_counter() to measure round-trip latency

# ========== GLOBAL CONSTANTS ==========

# These match the C++ server's filter parameters
FS = 100e3  # Sampling frequency: 100,000 Hz (100 kHz)
            # Scientific notation: 100e3 = 100 × 10^3 = 100,000
            # This is how many samples per second our signal has

B = 10e3    # Bandwidth: 10,000 Hz (10 kHz)
            # This is the cutoff frequency of the lowpass filter
            # Frequencies below 10kHz pass, above 10kHz get blocked

N = 16      # Half-length parameter
            # Filter will have 2*N = 32 taps (coefficients)
            # Determines filter length and frequency response sharpness

PORT = 8080  # UDP port number the server listens on
             # Must match the C++ server's PORT constant

SERVER_IP = "127.0.0.1"  # Server's IP address (localhost/loopback)
                         # Means "this computer" - packets never leave machine
                         # Could be changed to "192.168.1.5" for remote server


def calculate_correct_filter():
    """
    Calculate the theoretical FIR filter coefficients.
    
    This computes what the C++ server SHOULD be calculating.
    We use this as ground truth to verify the server's implementation.
    
    Returns:
        np.array: 32-element array of filter coefficients (unnormalized)
    """
    # Create array to store coefficients
    # np.zeros(size, dtype): creates array filled with zeros
    # 2 * N = 32 elements (our filter length)
    # dtype=np.float32: 32-bit floats (matches C++ 'float' type)
    h = np.zeros(2 * N, dtype=np.float32)

    # Calculate each coefficient
    for i in range(2 * N):
        n_val = i - N
        
        # Sinc function argument - we add 2 * pi for proper scaling. this normalizes the sinc to the bandwidth
        # in simple terms, the 2 * pi term converts from cycles to radians, which is needed for the sine function
        # sinc(x) = sin(pi*x)/(pi*x) is the normalized sinc function, which is how we ensure the sum of coefficients equals 1 after windowing
        # essentially, frequency to angular frequency conversion
        sinc_arg = 2 * np.pi * (B / FS) * (n_val + 0.5)
        if abs(sinc_arg) < 1e-10:
            sinc_val = 1.0
        else:
            sinc_val = np.sin(sinc_arg) / sinc_arg
        cos_window = 1.0 + np.cos(np.pi * (n_val + 0.5) / (N + 0.5))
        h[i] = sinc_val * cos_window
    h /= np.sum(h)
    return h


def test_impulse_response():
    """
    Test 1: Verify filter coefficients by sending an impulse.
    
    Theory: When you send an impulse δ[n] = [1, 0, 0, 0, ...] through
    a filter, the output IS the filter's impulse response, which equals
    the filter coefficients h[n].
    
    δ[n] * h[n] = h[n]
    
    So we send [1.0, 0, 0, ...] and expect to get back the coefficients.
    
    Returns:
        bool: True if test passes, False otherwise
    """

    print("\n=== TEST 1: IMPULSE RESPONSE (Filter Coefficients) ===")
    
    # ===== STEP 1: CALCULATE EXPECTED RESULT =====
    
    # Get theoretical coefficients (what server should calculate)
    h_expected = calculate_correct_filter()
    
    # Print expected values for comparison
    print(f"\n[EXPECTED] Filter coefficients h[n] (UNNORMALIZED):")
    
    # len(h_expected): number of elements in array (should be 32)
    print(f"  Length: {len(h_expected)}")
    
    # np.sum(): add all elements together
    # For unnormalized filter, sum ≠ 1.0 (probably around 31-32)
    # :.10f means "float with 10 decimal places"
    print(f"  Sum: {np.sum(h_expected):.10f}")
    
    # Show first 5 coefficients: h[0], h[1], h[2], h[3], h[4]
    # Array slicing: h_expected[:5] means "elements 0 through 4"
    print(f"  First 5:  {h_expected[:5]}")
    
    # Show last 5 coefficients: h[27], h[28], h[29], h[30], h[31]
    # Negative indexing: h[-5:] means "last 5 elements"
    print(f"  Last 5:   {h_expected[-5:]}")
    
    # ===== STEP 2: CREATE IMPULSE SIGNAL =====
    
    # Create impulse: [1.0, 0, 0, 0, ...]
    # 50 samples total (more than filter length of 32)
    impulse = np.zeros(50, dtype=np.float32)  # All zeros
    impulse[0] = 1.0  # Set first sample to 1.0
    
    # Now impulse = [1.0, 0.0, 0.0, ..., 0.0] (50 elements)
    
    # ===== STEP 3: CREATE UDP SOCKET =====
    
    # socket.socket(): create new socket
    # socket.AF_INET: Address Family = IPv4
    # socket.SOCK_DGRAM: Socket Type = UDP (datagram)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Set timeout: if no response after 2 seconds, raise exception
    # Prevents hanging forever if server is down
    sock.settimeout(2.0)  # 2.0 seconds
    
     # try-except-finally block for error handling
    try:
        # Start timer to measure round-trip time
        # time.perf_counter(): high-resolution timer (nanosecond precision)
        start = time.perf_counter()
        
        # ===== PACK DATA =====
        # struct.pack(): convert Python values to binary bytes
        # Format string: f'{len(impulse)}f'
        #   len(impulse) = 50
        #   So format is '50f' = "50 floats"
        # *impulse: unpacks array into individual arguments
        #   Like calling pack('50f', impulse[0], impulse[1], ..., impulse[49])
        # Returns: bytes object (200 bytes = 50 floats × 4 bytes each)
        data = struct.pack(f'{len(impulse)}f', *impulse)
        
        # ===== SEND TO SERVER =====
        # sock.sendto(data, address): send UDP packet
        # data: the 200 bytes we just packed
        # (SERVER_IP, PORT): destination address tuple
        #   = ('127.0.0.1', 8080)
        # This creates UDP packet and sends it to server
        sock.sendto(data, (SERVER_IP, PORT))
        
        # ===== RECEIVE RESPONSE =====
        # sock.recvfrom(bufsize): receive UDP packet
        # 4096: maximum bytes to receive (buffer size)
        # This BLOCKS (waits) until:
        #   1. Packet arrives, OR
        #   2. Timeout (2 sec) expires
        # Returns tuple: (data, address)
        #   data: bytes received
        #   address: sender's (IP, port) - we ignore with '_'
        response, _ = sock.recvfrom(4096)
        
        # Stop timer and calculate elapsed time
        # time.perf_counter() returns current time in seconds (float)
        # (end - start) gives seconds as float
        # × 1000 converts to milliseconds
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # ===== UNPACK RESPONSE =====
        # struct.unpack(): convert binary bytes back to Python values
        # Format: f'{len(impulse)}f' = '50f' = "50 floats"
        # response: the bytes received from server
        # Returns: tuple of 50 floats
        # np.array(): convert tuple to numpy array for easier manipulation
        received = np.array(struct.unpack(f'{len(impulse)}f', response))
        
        # Extract first 32 samples (the impulse response = coefficients)
        # received[:32] means "elements 0 through 31"
        # Remaining samples [32:50] should all be zero
        h_actual = received[:32]
        
        # ===== PRINT ACTUAL RESULTS =====
        print(f"\n[ACTUAL] From server (impulse response):")
        print(f"  Length: {len(h_actual)}")
        print(f"  Sum: {np.sum(h_actual):.10f}")
        print(f"  First 5:  {h_actual[:5]}")
        print(f"  Last 5:   {h_actual[-5:]}")
        
        # ===== COMPARE EXPECTED VS ACTUAL =====
        
        # Calculate element-wise absolute difference
        # np.abs(): absolute value of each element
        # diff[i] = |h_actual[i] - h_expected[i]|
        diff = np.abs(h_actual - h_expected)
        
        # Find maximum error across all coefficients
        # np.max(): returns largest value in array
        max_err = np.max(diff)
        
        # Calculate Root Mean Square Error
        # RMSE = sqrt(mean(diff^2))
        # Measures "average" error magnitude
        # diff**2: square each element
        # np.mean(): average of all elements
        # np.sqrt(): square root
        rmse = np.sqrt(np.mean(diff**2))
        
        print(f"\n[COMPARISON]:")
        # Scientific notation: .6e means "6 decimal places in exponential form"
        # Example: 0.000001234 → 1.234000e-06
        print(f"  Max error: {max_err:.6e}")
        print(f"  RMSE: {rmse:.6e}")
        
        # Check if arrays are "close enough"
        # np.allclose(a, b, atol=tolerance): True if all elements satisfy:
        #   |a[i] - b[i]| <= tolerance
        # atol=1e-6 means tolerance of 0.000001 (one millionth)
        close = np.allclose(h_actual, h_expected, atol=1e-6)
        print(f"  Match (tolerance 1e-6)? {close}")
        
        # Print timing
        # :.2f means "2 decimal places"
        print(f"  Time: {elapsed_ms:.2f} ms")
        
        # ===== DETERMINE PASS/FAIL =====
        if np.allclose(h_actual, h_expected, atol=1e-6):
            print(f"\nPASS: Impulse response matches filter coefficients")
            return True  # Test passed
        else:
            print(f"\nFAIL: Impulse response does not match")
            return False  # Test failed
            
    # ===== ERROR HANDLING =====        
    except Exception as e:
        # Exception: base class for all errors
        # 'as e': captures exception object
        # Catches ANY error (timeout, connection refused, etc.)
        print(f"\nERROR: {e}")
        return False  # Test failed due to error
    
    # ===== CLEANUP =====
    finally:
        # finally: ALWAYS executes (even if exception occurred)
        # Ensures socket is closed to free resources
        sock.close()

# ========== TEST 2: DIRECT CONVOLUTION ==========
def test_direct_convolution():
    """
    Test 2: Verify filtering accuracy with known signal.

    We generate a sine wave at 49099 Hz, send it through the filter,
    and compare the output to what NumPy's convolution gives us.
    
    This tests that the C++ convolution implementation is correct.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n=== TEST 2: DIRECT CONVOLUTION OUTPUT ===")
    
    # ===== STEP 1: GENERATE TEST SIGNAL =====

    FC_TEST = 49099 # Test frequency: 49099 Hz
                    # well above cutoff (10kHz) to see attenuation

    duration = 0.005 # Signal duration: 5 milliseconds
                     # Short signal to keep test quick
    

    # Generate time array
    # np.arange(start, stop, step): creates array [start, start+step, start+2*step, ...]
    # start=0, stop=0.005, step=1/FS=1/100000=0.00001
    # Result: [0.00000, 0.00001, 0.00002, ..., 0.00499]
    # Length: 0.005 / 0.00001 = 500 samples
    t = np.arange(0, duration, 1/FS)


    # Generate sine wave
    # np.sin(): sine function (element-wise on array)
    # 2 * np.pi * FC_TEST * t: argument to sine (phase)
    #   At t=0: phase=0, sin(0)=0
    #   At t=1/FC_TEST: phase=2π, sin(2π)=0 (one complete cycle)
    # .astype(np.float32): convert to 32-bit float (match C++)
    # x contains 500 samples of 49099 Hz sine wave
    x = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    # Print signal info
    print(f"\n[INPUT SIGNAL]:")
    print(f"  Frequency: {FC_TEST} Hz")
    # duration*1000: convert seconds to milliseconds
    # :.2f: 2 decimal places
    print(f"  Duration: {duration*1000:.2f} ms")
    print(f"  Length: {len(x)} samples")
    
    # ===== STEP 2: CALCULATE EXPECTED OUTPUT =====
    
    # Get filter coefficients
    h = calculate_correct_filter()  # 32 coefficients
    
    # Pad input with zeros at the beginning
    # Why? The C++ filter starts with delay_line = [0, 0, ..., 0]
    # So we need to simulate that initial zero state
    # np.concatenate([array1, array2]): join arrays end-to-end
    # np.zeros(31): 31 zeros (filter length - 1)
    # Result: [0,0,...,0, x[0], x[1], ..., x[499]]
    #         └ 31 zeros ┘└──── 500 samples ────┘
    # Total length: 31 + 500 = 531
    x_padded = np.concatenate([np.zeros(len(h)-1, dtype=np.float32), x])
    
    # Perform convolution
    # np.convolve(signal, filter, mode='valid'):
    #   Computes y[n] = Σ h[k] × x[n-k]
    #   mode='valid': only output where filter fully overlaps signal
    #   Returns: 531 - 32 + 1 = 500 samples (same length as original x)
    # This is EXACTLY what C++ does (causal filtering with zero initial state)
    y_expected_full = np.convolve(x_padded, h, mode='valid').astype(np.float32)
    
    print(f"\n[EXPECTED OUTPUT (causal convolution)]:")
    print(f"  Length: {len(y_expected_full)}")  # Should be 500
    print(f"  First 5: {y_expected_full[:5]}")  # First 5 samples
    # [32:37] means samples 32 through 36, which we choose to show because these are after the transient
    # After 32 samples (filter length), transient should settle
    # transient: initial distortion before steady-state. this distortion occurs because the filter needs time to "fill" its delay line
    # steady-state: filter output stabilizes, it now has full history of input samples
    print(f"  First 5 after transient: {y_expected_full[32:37]}")
    
    # ===== STEP 3: GET ACTUAL OUTPUT FROM SERVER =====
    
    # Create new socket for this test
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Start timer
        start = time.perf_counter()
        
        # Pack signal into bytes
        # x has 500 samples, so format is '500f'
        # Results in 2000 bytes (500 × 4)
        data = struct.pack(f'{len(x)}f', *x)
        
        # Send to server
        sock.sendto(data, (SERVER_IP, PORT))
        
        # Receive filtered result
        # Server should send back same number of samples (500)
        response, _ = sock.recvfrom(4096)

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Unpack response
        # Should get 500 floats back
        y_actual = np.array(struct.unpack(f'{len(x)}f', response))
        
        print(f"\n[ACTUAL OUTPUT:]")
        print(f"  Length: {len(y_actual)}")
        print(f"  First 5: {y_actual[:5]}")
        print(f"  First 5 after transient: {y_actual[32:37]}")
        print(f"  Time: {elapsed_ms:.2f} ms")
        
        # ===== STEP 4: COMPARE EXPECTED VS ACTUAL =====
        
        # Ensure arrays are same length (defensive programming)
        # min(): returns smaller of two values
        min_len = min(len(y_actual), len(y_expected_full))
        
        # Truncate both arrays to same length
        # [:min_len] means "elements 0 through min_len-1"
        y_actual_cmp = y_actual[:min_len]
        y_expected_cmp = y_expected_full[:min_len]
        
        # Calculate differences
        diff = np.abs(y_actual_cmp - y_expected_cmp)
        max_err = np.max(diff)
        rmse = np.sqrt(np.mean(diff**2))
        
        print(f"\n[FULL COMPARISON]:")
        print(f"  Length compared: {min_len}")
        print(f"  Max error: {max_err:.6e}")
        print(f"  RMSE: {rmse:.6e}")

        # Check if arrays are close enough
        print(f"  Match (tolerance 1e-4)? {np.allclose(y_actual_cmp, y_expected_cmp, atol=1e-4)}")
        
        # ===== STEP 5: ALSO CHECK STEADY-STATE =====
        # Initial samples have transient behavior
        # Let's also compare after filter has "warmed up"
        
        skip_samples = len(h)  # Skip first 32 samples (filter length)
        
        # Slice arrays to exclude transient
        # [skip_samples:] means "from element 32 to end"
        y_actual_steady = y_actual[skip_samples:]
        y_expected_steady = y_expected_full[skip_samples:]

        # Get length of steady-state region
        min_len_steady = min(len(y_actual_steady), len(y_expected_steady))
        
        # Only compare if there are steady-state samples
        if min_len_steady > 0:
            # Calculate errors for steady-state only
            diff_steady = np.abs(y_actual_steady[:min_len_steady] - y_expected_steady[:min_len_steady])
            max_err_steady = np.max(diff_steady)
            rmse_steady = np.sqrt(np.mean(diff_steady**2))
            
            print(f"\n[STEADY-STATE COMPARISON (skipping first {skip_samples})]:")
            print(f"  Length compared: {min_len_steady}")
            print(f"  Max error: {max_err_steady:.6e}")
            print(f"  RMSE: {rmse_steady:.6e}")
        
        # ===== STEP 6: DETERMINE PASS/FAIL =====
        if np.allclose(y_actual_cmp, y_expected_cmp, atol=1e-4):
            print(f"\nPASS: Convolution output matches expected")
            return True
        else:
            print(f"\nFAIL: Convolution output does not match")

            # Print detailed comparison for debugging
            print(f"\n[DEBUG] First 10 samples side-by-side:")
            # range(min(10, min_len)): iterate up to 10 times, or less if array shorter
            for i in range(min(10, min_len)):
                # :12.8f means "12 chars wide, 8 decimal places"
                # Aligns decimal points vertically for easy comparison
                print(f"  [{i}] Expected: {y_expected_cmp[i]:12.8f}, Actual: {y_actual_cmp[i]:12.8f}, Diff: {diff[i]:12.8f}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")

        # import traceback: module for detailed error info
        # traceback.print_exc(): prints full error traceback
        # Shows exactly where error occurred (line numbers, call stack)
        # traceback goes into more detail than just 'print(e)' - it shows context, as well as the chain of function calls 
        # that led to the error. by context, i mean it shows which functions were called, 
        # in what order, and where the error happened in that sequence.
        import traceback
        traceback.print_exc()
        return False
    finally:
        sock.close()

# ========== TEST 3: PASSBAND RESPONSE ==========
def test_passband_response():
    """
    Test 3: Verify filter passes low frequencies.
    
    The passband (0 - 10kHz) should let signals through with minimal
    attenuation. We test three frequencies well within the passband:
    500 Hz, 1 kHz, and 5 kHz.
    
    For a normalized filter, we'd expect 0 dB (no change).
    For unnormalized, we expect positive gain (amplification).
    
    Returns:
        bool: True if all passband frequencies pass through adequately
    """
    print("\n=== TEST 3: PASSBAND FREQUENCY RESPONSE ===")

    # the passband is frequencies below the cutoff (10 kHz).
    # we expect minimal attenuation for these frequencies.

    # high level summary:
    # we send sine waves at 500 Hz, 1 kHz, and 5 kHz (which are all below 10 kHz).
    # We expect minimal attenuation for frequencies within the passband.
    # for a normalized filter, we expect close to 0 dB attenuation.
    # for unnormalized, we expect positive gain (amplification).

    # Test these frequencies (all well below 10 kHz cutoff)
    freqs = [500, 1000, 5000]  # Hz
    
    # List to store results: [(freq1, attenuation1), (freq2, attenuation2), ...]
    results = []
    
    # Loop through each test frequency
    # 'for variable in list:' iterates over list elements
    for fc in freqs:
        # Generate 10ms of signal at this frequency
        duration = 0.01

        # Time array: [0, 0.00001, 0.00002, ..., 0.00999]
        # Length: 0.01 / 0.00001 = 1000 samples
        t = np.arange(0, duration, 1/FS)
        
        # Generate sine wave at test frequency
        # fc changes each loop iteration: 500, 1000, 5000
        x = np.sin(2 * np.pi * fc * t).astype(np.float32)
        
        # Create new socket for each test (clean state)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        
        try:
            # Start timer
            start = time.perf_counter()
            
            # Pack signal (1000 floats = 4000 bytes)
            data = struct.pack(f'{len(x)}f', *x)
            
            # Send to server
            sock.sendto(data, (SERVER_IP, PORT))
            
            # Receive filtered output
            response, _ = sock.recvfrom(4096)
            
            # Calculate elapsed time
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Unpack response
            y = np.array(struct.unpack(f'{len(x)}f', response))
            
            # ===== CALCULATE ATTENUATION =====
            
            # RMS (Root Mean Square) = measure of signal amplitude
            # RMS = sqrt(mean(signal^2))
            # x**2: square each sample
            # np.mean(): average
            # np.sqrt(): square root
            # For sine wave: RMS ≈ 0.707 × peak amplitude
            input_rms = np.sqrt(np.mean(x**2))
            output_rms = np.sqrt(np.mean(y**2))
            
            # Calculate attenuation in decibels (dB)
            # dB = 20 × log10(output/input)
            # If output = input: 20×log10(1) = 0 dB (no change)
            # If output = 2×input: 20×log10(2) = +6 dB (amplification)
            # If output = 0.5×input: 20×log10(0.5) = -6 dB (attenuation)
            # np.log10(): base-10 logarithm
            attenuation_db = 20 * np.log10(output_rms / input_rms)
            
            # Print result for this frequency
            # :5d means "integer, 5 chars wide, right-aligned"
            # :6.2f means "float, 6 chars wide, 2 decimal places"
            print(f"\n  {fc:5d} Hz: {attenuation_db:6.2f} dB, time: {elapsed_ms:.2f} ms")
            
            # Store result as tuple
            # .append(): add element to end of list
            results.append((fc, attenuation_db))
            
        except Exception as e:
            # If this frequency test fails, print error but continue to next
            print(f"  {fc:5d} Hz: ERROR - {e}")
        finally:
            # cleanup socket
            sock.close()
    
    # ===== EVALUATE RESULTS =====
    # NOTE: With normalization, we expect attenuation close to 0 dB.
    # Without normalization, we expect positive gain (amplification).
    # Here, we simply check if attenuation is less than 3 dB.
    # Check if passband (should be < 3 dB attenuation)

    # 'all(condition for items)': True if condition true for ALL items
    # 'for _, att in results': _ ignores frequency, att is attenuation
    # 'att < 3': attenuation less than 3 dB (not too attenuated)
    passband_ok = all(abs(att) < 3 for _, att in results)
    # we choose 3 dB as threshold because it's a common engineering standard for acceptable passband ripple 
        # (according to the 3 dB point definition).
    
    if passband_ok:
        print(f"\nPASS: Passband response OK (all < 3dB)")
        return True
    else:
        print(f"\nFAIL: Passband response not OK")
        return False

# ========== TEST 4: STOPBAND RESPONSE ==========
def test_stopband_response():
    """
    Test the stopband frequency response of the FIR filter.
    """

    # the stopband is frequencies above the cutoff (10 kHz).
    # the cutoff is at 10 kHz, so we test frequencies well above that.
    # we get the cutoff from the global constant B.

    # high level summary:
    # we send sine waves at 15 kHz, 25 kHz, and 40 kHz.
    # we measure how much the filter attenuates these signals.
    # we expect significant attenuation (large negative dB values).

    print("\n=== TEST 4: STOPBAND FREQUENCY RESPONSE ===")    

    # Frequencies to test (all above 10 kHz cutoff)
    freqs = [15000, 25000, 40000]
    results = []
    
    for fc in freqs:
        # Generate 10ms of signal at this frequency
        duration = 0.01

        # Time array: [0, 0.00001, 0.00002, ..., 0.00999]
        # Length: 0.01 / 0.00001 = 1000 samples
        t = np.arange(0, duration, 1/FS)

        # Generate sine wave at test frequency
        # fc changes each loop iteration: 15000, 25000, 40000
        x = np.sin(2 * np.pi * fc * t).astype(np.float32)
        
        # Create new socket for each test
        # socket.socket(): create UDP socket
        # socket.AF_INET: IPv4
        # socket.SOCK_DGRAM: UDP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # sock.setsockopt(): set socket options
        # Here we set a timeout of 2 seconds
        sock.settimeout(2.0)
        
        try:
            # Start timer
            start = time.perf_counter()

            # Pack signal into bytes
            # struct.pack(): convert array of floats to bytes
            data = struct.pack(f'{len(x)}f', *x)

            # Send to server
            sock.sendto(data, (SERVER_IP, PORT))

            # Receive filtered output
            response, _ = sock.recvfrom(4096)

            # Calculate elapsed time 
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Unpack response
            # struct.unpack(): convert bytes back to array of floats
            y = np.array(struct.unpack(f'{len(x)}f', response))
            
            # Calculate RMS of input and output signals
            # RMS = sqrt(mean(signal^2))
            input_rms = np.sqrt(np.mean(x**2))
            output_rms = np.sqrt(np.mean(y**2))

            # Calculate attenuation in decibels (dB)
            # dB = 20 × log10(output/input)
            # we don't have the special case of output_rms ≈ 0 here because output_rms is expected to be small
            # actually, that doesn't make sense - we should still handle the case where output_rms is very small to avoid log10(0)
            if output_rms < 1e-6:
                attenuation_db = -100.0  # Assign large negative value for practical purposes
            else:
                 # Normal case: calculate actual attenuation
                attenuation_db = 20 * np.log10(output_rms / input_rms)
                # attenuation is how much signal is reduced (negative dB). 
                # E.g., -20 dB means output is 1/10th the amplitude of input
                # for normalized filter, expect large negative value
                # for unnormalized, expect still negative but less extreme


            print(f"\n  {fc:5d} Hz: {attenuation_db:6.2f} dB, time: {elapsed_ms:.2f} ms")
            results.append((fc, attenuation_db))
            
        except Exception as e:
            print(f"  {fc:5d} Hz: ERROR - {e}")
        finally:
            sock.close()
    
    # ===== EVALUATE RESULTS =====

    # Check if stopband 
    # 'all(condition for items)': True if condition true for ALL items
    # 'for _, att in results': _ ignores frequency, att is attenuation
    # 'att < -10': attenuation less than -10 dB (significant reduction). -10 dB means output is 1/3.16 of input amplitude
    stopband_ok = all(att < -10 for _, att in results)
    if stopband_ok:
        print(f"\nPASS: Stopband response OK (all > 10dB attenuation)")
        return True
    else:
        print(f"\nFAIL: Stopband response not OK")
        return False

# ========== TEST 5: PACKET CONTINUITY ==========
def test_packet_continuity():
    """
        This is THE critical test for streaming. We send a continuous signal
    in TWO separate packets and verify there's no discontinuity at the
    boundary.

    If the filter resets between packets, we'd see a huge jump.
    If the filter maintains state correctly, the output should be smooth.

    Returns:
        bool: True if continuity is maintained across packet boundary
    """

    print("=== TEST 5: PACKET CONTINUITY ===")
    
    # ===== STEP 1: GENERATE CONTINUOUS SIGNAL =====

    FC_TEST = 1000  # Test frequency: 1 kHz
    duration = 0.02  # Total duration: 20 milliseconds

    # Generate time array
    # np.arange(start, stop, step): creates array [start, start+step, start+2*step, ...]
    # start=0, stop=0.02, step=1/FS=1/100000=0.00001
    # Result: [0.00000, 0.00001, 0.00002, ..., 0.01999]
    # Length: 0.02 / 0.00001 = 2000 samples 
    t = np.arange(0, duration, 1/FS)

    # Generate sine wave
    # np.sin(): sine function (element-wise on array)
    # 2 * np.pi * FC_TEST * t: argument to sine (phase)
    #   At t=0: phase=0, sin(0)=0
    #   At t=1/FC_TEST: phase=2π, sin(2π)=0 (one complete cycle)
    # .astype(np.float32): convert to 32-bit float (match C++)
    # signal contains 2000 samples of 1 kHz sine wave
    signal = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    # ===== STEP 2: CREATE SOCKET =====
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
     # ===== IMPORTANT: USE connect() =====
    # sock.connect(): "lock" this socket to one destination
    # After this, we can use send() instead of sendto()
    # Slightly more convenient for multiple transmissions
    # send vs sendto:
    #   send(data): sends to connected address
    #   sendto(data, address): specify address each time

    # we did this in the unormalized version, but we didnt do it here because we forgot - my bad!
    # sock.connect((SERVER_IP, PORT))

    try:
        # ===== STEP 3: SPLIT SIGNAL INTO TWO CHUNKS =====
        # First chunk: samples 0-99 (100 samples)
        # Array slicing: [:100] means "from start up to (not including) 100"
        chunk1 = signal[:100]
        
        # Second chunk: samples 100-199 (100 samples)
        # [100:200] means "from 100 up to (not including) 200"
        chunk2 = signal[100:200]
        
        # These chunks are CONSECUTIVE parts of the same signal
        # The last sample of chunk1 (signal[99]) comes RIGHT BEFORE
        # the first sample of chunk2 (signal[100])

        # Start timer for total round-trip of BOTH packets
        start = time.perf_counter()

        # ===== STEP 4: SEND FIRST CHUNK =====
        # Pack first chunk into bytes (100 floats = 400 bytes)
        data1 = struct.pack(f'{len(chunk1)}f', *chunk1)
        # Send to server
        sock.sendto(data1, (SERVER_IP, PORT))
        # Receive response for chunk1
        response1, _ = sock.recvfrom(4096)
        # Unpack response (100 floats)
        out1 = np.array(struct.unpack(f'{len(chunk1)}f', response1))
        
        # ===== STEP 5: SEND SECOND CHUNK =====
        # Pack second chunk into bytes (100 floats = 400 bytes)
        data2 = struct.pack(f'{len(chunk2)}f', *chunk2)
        # Send to server
        sock.sendto(data2, (SERVER_IP, PORT))
        # Receive response for chunk2
        response2, _ = sock.recvfrom(4096)
        # Unpack response (100 floats)
        out2 = np.array(struct.unpack(f'{len(chunk2)}f', response2))

        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # ===== STEP 6: CHECK BOUNDARY CONTINUITY =====

        # The last sample of out1 should be very close to
        # the first sample of out2 (they're consecutive in time!)
        
        # out1[-1]: last element of first chunk's output
        # out2[0]: first element of second chunk's output
        # abs(): absolute value (ignore whether jump is up or down)
        boundary_jump = abs(out2[0] - out1[-1])
        
        # For a continuous signal through a continuous filter,
        # this jump should be VERY small (< 0.05 for normalized filter)
        # for a normalized filter, we expect even smaller jump - ideally close to 0.
        # this threshold of 0.05 is chosen based on expected signal levels and filter behavior.
        # this is because the output signal amplitude is expected to be within [-1, 1] range for normalized filters.

        # print(f"\n[PACKET 1]:")
        # print(f"  Length: {len(chunk1)}")
        # print(f"  Output last 3: {out1[-3:]}")
        
        # print(f"\n[PACKET 2]:")
        # print(f"  Length: {len(chunk2)}")
        # print(f"  Output first 3: {out2[:3]}")
        
        # print(f"\n[BOUNDARY]:")
        # print(f"  Jump (out2[0] - out1[-1]): {boundary_jump:.6f}")
        # print(f"  Time: {elapsed_ms:.2f} ms")
        
        # ===== STEP 7: EVALUATE CONTINUITY =====
        # If jump is less than 0.05, we consider continuity maintained
        # Otherwise, it's a failure
        if boundary_jump < 0.05:
            print(f"\nPASS: Continuity maintained (jump < 0.05)")
            return True
        else:
            print(f"\nFAIL: Discontinuity detected (jump >= 0.05)")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        sock.close()

    # high level summary:
    # If the filter maintains state correctly across packets,
    # the output, which is a continuous sine wave,
    # should be smooth with a small jump at the boundary.
    # If the filter resets between packets, there will be a large jump between the values of out1[-1] and out2[0], which are
    # supposed to be consecutive samples of the same continuous signal.


if __name__ == "__main__":
    print("\nC++ server should be on port 8080!")
    print("Press Enter to start tests:")
    input()
    
    results = []
    results.append(("Impulse Response", test_impulse_response()))
    results.append(("Direct Convolution", test_direct_convolution()))
    results.append(("Passband Response", test_passband_response()))
    results.append(("Stopband Response", test_stopband_response()))
    results.append(("Packet Continuity", test_packet_continuity()))
        
    print("\n\nTEST SUMMARY\n")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:30s} {status}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
