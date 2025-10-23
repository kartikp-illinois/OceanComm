#!/usr/bin/env python3

import socket
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import traceback
from typing import List, Tuple
from datetime import datetime
import os

# Filter and signal parameters
FS = 100e3      # Sampling frequency
B = 10e3        # Filter bandwidth  
N = 16          # Filter half-length
FC_TEST = 5e3   # Test frequency (within passband)
FC_TEST2 = 25e3 # Test frequency (outside passband)
PORT = 8080
SERVER_IP = "127.0.0.1"

# Debug flag
DEBUG = True

# Output file path
RESULTS_DIR = "test_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(RESULTS_DIR, f"fir_test_results_{TIMESTAMP}.txt")
DEBUG_FILE = os.path.join(RESULTS_DIR, f"fir_test_debug_{TIMESTAMP}.txt")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

class DualLogger:
    """Log to both console and file simultaneously"""
    def __init__(self, console_file, debug_file):
        self.console_file = open(console_file, 'w')
        self.debug_file = open(debug_file, 'w')
    
    def print(self, msg, level="INFO"):
        """Print to console and log file"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_msg = f"[{timestamp}] {msg}"
        print(formatted_msg)
        self.console_file.write(formatted_msg + "\n")
        self.console_file.flush()
    
    def debug(self, msg):
        """Print debug message to console, console log, and debug log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if DEBUG:
            debug_msg = f"[{timestamp}] [DEBUG] {msg}"
            print(debug_msg)
            self.console_file.write(debug_msg + "\n")
            self.debug_file.write(debug_msg + "\n")
            self.console_file.flush()
            self.debug_file.flush()
    
    def section(self, msg):
        """Print a section header"""
        separator = "=" * 80
        self.print(separator)
        self.print(msg)
        self.print(separator)
    
    def subsection(self, msg):
        """Print a subsection header"""
        separator = "-" * 80
        self.print(separator)
        self.print(msg)
        self.print(separator)
    
    def close(self):
        """Close all file handles"""
        self.console_file.close()
        self.debug_file.close()

# Initialize logger
logger = DualLogger(RESULTS_FILE, DEBUG_FILE)

def generate_test_signal(fc, duration_sec, fs):
    """Generate test sinusoidal signal x[n] = sin(2*pi*Fc*n/Fs)"""
    logger.debug(f"Generating test signal: fc={fc}, duration={duration_sec}s, fs={fs}")
    t = np.arange(0, duration_sec, 1/fs)
    signal = np.sin(2 * np.pi * fc * t).astype(np.float32)
    logger.debug(f"Generated {len(signal)} samples, range: [{np.min(signal):.6f}, {np.max(signal):.6f}]")
    return signal

def calculate_expected_filter(fs, b, n):
    """Calculate expected (causal, normalized) filter coefficients for verification"""
    logger.debug(f"Computing expected filter coefficients: fs={fs}, b={b}, n={n}")
    
    h_centered = np.zeros(2 * n, dtype=np.float32)
    
    logger.debug(f"Creating {2*n} centered coefficients...")
    for i in range(2 * n):
        n_val = i - n  # index from -N .. N-1
        
        sinc_arg = (b / fs) * (n_val + 0.5)
        if abs(sinc_arg) < 1e-10:
            sinc_val = 1.0
        else:
            sinc_val = np.sin(np.pi * sinc_arg) / (np.pi * sinc_arg)
        
        cos_window = 1.0 + np.cos(np.pi * (n_val + 0.5) / (n + 0.5))
        h_centered[i] = sinc_val * cos_window
        
        if i < 5:
            logger.debug(f"  h_centered[{i}] (n={n_val}): sinc_arg={sinc_arg:.6f}, sinc_val={sinc_val:.6f}, cos_window={cos_window:.6f}, h={h_centered[i]:.6f}")

    # normalize to DC gain = 1 (match C++ normalization)
    sum_h_before = np.sum(h_centered)
    logger.debug(f"Sum before normalization: {sum_h_before:.8f}")
    
    if abs(sum_h_before) > 1e-9:
        h_centered /= sum_h_before
        logger.debug(f"Normalized by dividing by {sum_h_before:.8f}")
    else:
        logger.debug(f"WARNING: Sum too small, skipping normalization")

    sum_h_after = np.sum(h_centered)
    logger.debug(f"Sum after normalization: {sum_h_after:.8f}")

    # rotate so index 0 corresponds to h[0] (causal ordering used by C++ process)
    h_causal = np.zeros_like(h_centered)
    L = 2 * n
    logger.debug(f"Rotating indices to causal ordering (L={L})...")
    
    for k in range(L):
        src = (k + n) % L
        h_causal[k] = h_centered[src]
    
    logger.debug(f"First 5 causal coefficients: {h_causal[:5]}")
    logger.debug(f"Last 5 causal coefficients: {h_causal[-5:]}")
    logger.debug(f"Causal filter range: [{np.min(h_causal):.8f}, {np.max(h_causal):.8f}]")
    
    return h_causal

def compute_expected_convolution(signal: np.ndarray, h_filter: np.ndarray) -> np.ndarray:
    """Compute expected convolution output with full state maintenance"""
    logger.debug(f"Computing expected convolution: signal_len={len(signal)}, filter_len={len(h_filter)}")
    
    output = np.zeros_like(signal, dtype=np.float32)
    delay_line = np.zeros(len(h_filter), dtype=np.float32)
    write_idx = 0
    
    for n in range(len(signal)):
        # Store sample in delay line
        delay_line[write_idx] = signal[n]
        
        # Compute output using symmetric filter optimization
        y_n = 0.0
        idx1 = write_idx
        idx2 = (write_idx + N) % len(h_filter)
        
        for k in range(N):
            y_n += h_filter[k] * (delay_line[idx1] + delay_line[idx2])
            idx1 = (idx1 - 1) % len(h_filter)
            idx2 = (idx2 + 1) % len(h_filter)
        
        output[n] = y_n
        write_idx = (write_idx + 1) % len(h_filter)
        
        if n < 5:
            logger.debug(f"  y[{n}] = {y_n:.8f} (input={signal[n]:.6f})")
    
    logger.debug(f"Convolution output range: [{np.min(output):.8f}, {np.max(output):.8f}]")
    logger.debug(f"Convolution output RMS: {np.sqrt(np.mean(output**2)):.8f}")
    
    return output

def send_and_receive_udp(signal, server_ip, port, chunk_size=256) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Send signal via UDP and receive filtered response with detailed logging"""
    logger.debug(f"Starting UDP communication: server={server_ip}:{port}, chunk_size={chunk_size}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5.0)
    
    output_signal = []
    chunk_data = []
    
    try:
        logger.debug(f"Socket created, total signal length: {len(signal)} samples")
        
        # Send signal in chunks to simulate streaming
        chunk_num = 0
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i:i+chunk_size]
            logger.debug(f"\n--- Chunk {chunk_num} ---")
            logger.debug(f"Sending chunk: offset={i}, size={len(chunk)} samples")
            
            # Convert to bytes
            try:
                data = struct.pack(f'{len(chunk)}f', *chunk)
                logger.debug(f"Packed into {len(data)} bytes ({len(data)/4:.0f} floats)")
            except struct.error as e:
                logger.debug(f"ERROR packing chunk: {e}")
                raise
            
            # Send to server
            try:
                bytes_sent = sock.sendto(data, (server_ip, port))
                logger.debug(f"Sent {bytes_sent} bytes to {server_ip}:{port}")
                if bytes_sent != len(data):
                    logger.debug(f"WARNING: Sent {bytes_sent} bytes but expected {len(data)} bytes")
            except socket.error as e:
                logger.debug(f"ERROR sending data: {e}")
                raise
            
            # Receive response
            try:
                response, addr = sock.recvfrom(4096)
                logger.debug(f"Received {len(response)} bytes from {addr}")
                
                # Unpack floats
                if len(response) % 4 != 0:
                    logger.debug(f"ERROR: Response size {len(response)} is not multiple of 4")
                    raise ValueError(f"Response size mismatch")
                
                num_floats = len(response) // 4
                logger.debug(f"Unpacking {num_floats} floats from response")
                
                received_chunk = struct.unpack(f'{num_floats}f', response)
                logger.debug(f"Unpacked chunk: first 3 values = {received_chunk[:3]}, last 3 values = {received_chunk[-3:]}")
                logger.debug(f"Unpacked range: [{np.min(received_chunk):.8f}, {np.max(received_chunk):.8f}]")
                
                if len(received_chunk) != len(chunk):
                    logger.debug(f"ERROR: Length mismatch! Sent {len(chunk)} samples, got {len(received_chunk)} back")
                
                output_signal.extend(received_chunk)
                chunk_data.append((chunk, np.array(received_chunk)))
                
            except socket.timeout:
                logger.debug(f"ERROR: Socket timeout waiting for response")
                raise
            except struct.error as e:
                logger.debug(f"ERROR unpacking response: {e}")
                raise
            
            # Small delay to simulate real-time streaming
            time.sleep(0.001)
            chunk_num += 1
        
        output_array = np.array(output_signal)
        logger.debug(f"\nTotal output collected: {len(output_array)} samples")
        logger.debug(f"Output range: [{np.min(output_array):.8f}, {np.max(output_array):.8f}]")
        
        return output_array, chunk_data
    
    except Exception as e:
        logger.debug(f"EXCEPTION in send_and_receive_udp: {e}")
        traceback.print_exc()
        raise
    
    finally:
        sock.close()
        logger.debug("Socket closed")

def test_filter_response():
    """Test the FIR filter with comprehensive debug output"""
    logger.section("FIR Filter UDP Test Bench - Comprehensive Debug Version")
    
    # Test parameters
    duration = 0.1  # 100ms test signal
    
    # Generate test signals
    logger.print(f"Generating test signals...")
    logger.print(f"  - Sampling rate: {FS/1000:.0f} kHz")
    logger.print(f"  - Filter bandwidth: {B/1000:.0f} kHz")
    logger.print(f"  - Signal duration: {duration*1000:.0f} ms")
    logger.print(f"  - Filter half-length N: {N}")
    logger.print(f"  - Total filter length: {2*N}")
    
    # Pre-compute expected filter
    logger.subsection("Computing Expected Filter Coefficients")
    expected_h = calculate_expected_filter(FS, B, N)
    logger.print(f"Expected filter computed with {len(expected_h)} coefficients")
    
    # Test 1: Low frequency signal (should pass through)
    logger.subsection(f"TEST 1: {FC_TEST/1000:.0f} kHz sine wave (in passband)")
    signal1 = generate_test_signal(FC_TEST, duration, FS)
    
    try:
        logger.debug(f"Starting UDP communication for Test 1...")
        filtered1, chunks1 = send_and_receive_udp(signal1, SERVER_IP, PORT)
        
        logger.debug(f"Received {len(filtered1)} output samples")
        
        if len(filtered1) == len(signal1):
            logger.debug("✓ Output length matches input length")
            
            # Compute expected convolution
            logger.debug("Computing expected convolution locally...")
            expected_output1 = compute_expected_convolution(signal1, expected_h)
            
            # Calculate RMS to measure signal strength
            input_rms = np.sqrt(np.mean(signal1**2))
            output_rms = np.sqrt(np.mean(filtered1**2))
            expected_rms = np.sqrt(np.mean(expected_output1**2))
            
            logger.debug(f"Input RMS: {input_rms:.8f}")
            logger.debug(f"Output RMS (from server): {output_rms:.8f}")
            logger.debug(f"Expected RMS (computed): {expected_rms:.8f}")
            
            attenuation_db = 20 * np.log10(output_rms / input_rms) if input_rms > 0 else -np.inf
            logger.debug(f"Attenuation (dB): {attenuation_db:.4f}")
            
            # Compare output with expected
            mse = np.mean((filtered1 - expected_output1)**2)
            max_error = np.max(np.abs(filtered1 - expected_output1))
            correlation = np.corrcoef(filtered1, expected_output1)[0, 1]
            
            logger.debug(f"Comparison with expected:")
            logger.debug(f"  MSE: {mse:.2e}")
            logger.debug(f"  Max error: {max_error:.8f}")
            logger.debug(f"  Correlation: {correlation:.6f}")
            logger.debug(f"  First 5 actual vs expected: {filtered1[:5]} vs {expected_output1[:5]}")
            
            logger.print(f"  ✓ Input RMS: {input_rms:.8f}")
            logger.print(f"  ✓ Output RMS: {output_rms:.8f}")
            logger.print(f"  ✓ Expected RMS: {expected_rms:.8f}")
            logger.print(f"  ✓ Attenuation: {attenuation_db:.4f} dB")
            logger.print(f"  ✓ Correlation with expected: {correlation:.6f}")
            logger.print(f"  ✓ Max error from expected: {max_error:.8f}")
            
            if attenuation_db > -3:  # Less than 3dB attenuation in passband
                logger.print("  ✓ PASS: Signal in passband preserved")
            else:
                logger.print("  ✗ FAIL: Excessive attenuation in passband")
            
            if correlation > 0.99:
                logger.print("  ✓ PASS: Output matches expected convolution")
            else:
                logger.print("  ✗ FAIL: Output does not match expected convolution")
        else:
            logger.print(f"  ✗ FAIL: Length mismatch (expected {len(signal1)}, got {len(filtered1)})")
            logger.debug(f"CRITICAL: Output length mismatch!")
            
    except Exception as e:
        logger.print(f"  ✗ FAIL: {e}")
        logger.debug(f"EXCEPTION in Test 1: {e}")
        traceback.print_exc()
    
    # Test 2: High frequency signal (should be attenuated)
    logger.subsection(f"TEST 2: {FC_TEST2/1000:.0f} kHz sine wave (outside passband)")
    signal2 = generate_test_signal(FC_TEST2, duration, FS)
    
    try:
        logger.debug(f"Starting UDP communication for Test 2...")
        filtered2, chunks2 = send_and_receive_udp(signal2, SERVER_IP, PORT)
        
        logger.debug(f"Received {len(filtered2)} output samples")
        
        if len(filtered2) == len(signal2):
            logger.debug("✓ Output length matches input length")
            
            # Compute expected convolution
            logger.debug("Computing expected convolution locally...")
            expected_output2 = compute_expected_convolution(signal2, expected_h)
            
            input_rms = np.sqrt(np.mean(signal2**2))
            output_rms = np.sqrt(np.mean(filtered2**2))
            expected_rms = np.sqrt(np.mean(expected_output2**2))
            
            logger.debug(f"Input RMS: {input_rms:.8f}")
            logger.debug(f"Output RMS (from server): {output_rms:.8f}")
            logger.debug(f"Expected RMS (computed): {expected_rms:.8f}")
            
            attenuation_db = 20 * np.log10(output_rms / input_rms) if input_rms > 0 else -np.inf
            logger.debug(f"Attenuation (dB): {attenuation_db:.4f}")
            
            # Compare output with expected
            mse = np.mean((filtered2 - expected_output2)**2)
            max_error = np.max(np.abs(filtered2 - expected_output2))
            correlation = np.corrcoef(filtered2, expected_output2)[0, 1]
            
            logger.debug(f"Comparison with expected:")
            logger.debug(f"  MSE: {mse:.2e}")
            logger.debug(f"  Max error: {max_error:.8f}")
            logger.debug(f"  Correlation: {correlation:.6f}")
            logger.debug(f"  First 5 actual vs expected: {filtered2[:5]} vs {expected_output2[:5]}")
            
            logger.print(f"  ✓ Input RMS: {input_rms:.8f}")
            logger.print(f"  ✓ Output RMS: {output_rms:.8f}")
            logger.print(f"  ✓ Expected RMS: {expected_rms:.8f}")
            logger.print(f"  ✓ Attenuation: {attenuation_db:.4f} dB")
            logger.print(f"  ✓ Correlation with expected: {correlation:.6f}")
            logger.print(f"  ✓ Max error from expected: {max_error:.8f}")
            
            if attenuation_db < -20:  # At least 20dB attenuation outside passband
                logger.print("  ✓ PASS: Signal outside passband attenuated")
            else:
                logger.print("  ✗ FAIL: Insufficient attenuation outside passband")
            
            if correlation > 0.99:
                logger.print("  ✓ PASS: Output matches expected convolution")
            else:
                logger.print("  ✗ FAIL: Output does not match expected convolution")
        else:
            logger.print(f"  ✗ FAIL: Length mismatch (expected {len(signal2)}, got {len(filtered2)})")
            logger.debug(f"CRITICAL: Output length mismatch!")
            
    except Exception as e:
        logger.print(f"  ✗ FAIL: {e}")
        logger.debug(f"EXCEPTION in Test 2: {e}")
        traceback.print_exc()
    
    # Test 3: Impulse response
    logger.subsection(f"TEST 3: Impulse response")
    impulse = np.zeros(100, dtype=np.float32)
    impulse[0] = 1.0
    logger.debug(f"Created impulse signal: 100 samples, impulse at position 0")
    
    try:
        logger.debug(f"Starting UDP communication for Test 3...")
        impulse_response, chunks3 = send_and_receive_udp(impulse, SERVER_IP, PORT)
        
        logger.debug(f"Received {len(impulse_response)} output samples")
        
        if len(impulse_response) == len(impulse):
            logger.debug("✓ Output length matches input length")
            
            # The impulse response should match the filter coefficients (first 32 samples)
            actual_h = np.array(impulse_response[:32])
            
            logger.debug(f"First 32 response samples (should be filter coefficients): {actual_h}")
            logger.debug(f"Expected filter coefficients: {expected_h}")
            
            # Calculate correlation
            correlation = np.corrcoef(expected_h, actual_h)[0, 1]
            mse = np.mean((expected_h - actual_h)**2)
            max_error = np.max(np.abs(expected_h - actual_h))
            
            logger.debug(f"Impulse response comparison:")
            logger.debug(f"  Correlation: {correlation:.8f}")
            logger.debug(f"  MSE: {mse:.2e}")
            logger.debug(f"  Max error: {max_error:.8f}")
            logger.debug(f"  Expected first 10: {expected_h[:10]}")
            logger.debug(f"  Actual first 10: {actual_h[:10]}")
            
            logger.print(f"  ✓ Correlation with expected: {correlation:.8f}")
            logger.print(f"  ✓ Max error from expected: {max_error:.8f}")
            logger.print(f"  ✓ MSE: {mse:.2e}")
            
            if correlation > 0.9999:
                logger.print("  ✓ PASS: Impulse response matches expected coefficients")
            else:
                logger.print("  ✗ FAIL: Impulse response doesn't match expected")
        else:
            logger.print(f"  ✗ FAIL: Length mismatch (expected {len(impulse)}, got {len(impulse_response)})")
            logger.debug(f"CRITICAL: Output length mismatch!")
            
    except Exception as e:
        logger.print(f"  ✗ FAIL: {e}")
        logger.debug(f"EXCEPTION in Test 3: {e}")
        traceback.print_exc()
    
    # Test 4: Continuity across packets
    logger.subsection(f"TEST 4: Continuity across packet boundaries")
    long_signal = generate_test_signal(FC_TEST, 0.2, FS)  # 200ms signal
    logger.debug(f"Created long signal for continuity test: {len(long_signal)} samples")
    
    try:
        logger.debug(f"Starting UDP communication for Test 4 with small chunks...")
        filtered_chunked, chunks4 = send_and_receive_udp(long_signal, SERVER_IP, PORT, chunk_size=50)
        
        logger.debug(f"Received {len(filtered_chunked)} output samples in {len(chunks4)} chunks")
        
        if len(filtered_chunked) == len(long_signal):
            logger.debug("✓ Output length matches input length")
            
            # Compute expected convolution
            logger.debug("Computing expected convolution locally...")
            expected_output4 = compute_expected_convolution(long_signal, expected_h)
            
            # Check for discontinuities (sudden jumps in output)
            diff = np.diff(filtered_chunked)
            max_jump = np.max(np.abs(diff))
            mean_jump = np.mean(np.abs(diff))
            
            logger.debug(f"Sample-to-sample differences:")
            logger.debug(f"  Max jump: {max_jump:.8f}")
            logger.debug(f"  Mean jump: {mean_jump:.8f}")
            
            # Check at packet boundaries
            chunk_size = 50
            for i, (sent_chunk, recv_chunk) in enumerate(chunks4[:-1]):
                boundary_idx = len(recv_chunk) - 1
                jump_at_boundary = abs(chunks4[i+1][1][0] - recv_chunk[-1])
                logger.debug(f"  Jump at packet {i}/{i+1} boundary: {jump_at_boundary:.8f}")
            
            # Compare with expected
            mse = np.mean((filtered_chunked - expected_output4)**2)
            max_error = np.max(np.abs(filtered_chunked - expected_output4))
            correlation = np.corrcoef(filtered_chunked, expected_output4)[0, 1]
            
            logger.debug(f"Comparison with expected convolution:")
            logger.debug(f"  Correlation: {correlation:.6f}")
            logger.debug(f"  MSE: {mse:.2e}")
            logger.debug(f"  Max error: {max_error:.8f}")
            
            logger.print(f"  ✓ Maximum sample-to-sample jump: {max_jump:.8f}")
            logger.print(f"  ✓ Mean sample-to-sample jump: {mean_jump:.8f}")
            logger.print(f"  ✓ Correlation with expected: {correlation:.6f}")
            logger.print(f"  ✓ Max error from expected: {max_error:.8f}")
            
            if max_jump < 0.1:  # Reasonable threshold for continuity
                logger.print("  ✓ PASS: No significant discontinuities detected")
            else:
                logger.print("  ✗ FAIL: Large discontinuities detected")
            
            if correlation > 0.99:
                logger.print("  ✓ PASS: Output matches expected across packets")
            else:
                logger.print("  ✗ FAIL: Output doesn't match expected")
        else:
            logger.print(f"  ✗ FAIL: Length mismatch (expected {len(long_signal)}, got {len(filtered_chunked)})")
            logger.debug(f"CRITICAL: Output length mismatch!")
            
    except Exception as e:
        logger.print(f"  ✗ FAIL: {e}")
        logger.debug(f"EXCEPTION in Test 4: {e}")
        traceback.print_exc()
    
    logger.section("Test bench completed")


if __name__ == "__main__":
    try:
        logger.section("FIR Filter UDP Test Bench - Debug Version")
        logger.print("Make sure the C++ UDP server is running on port 8080")
        logger.print(f"Results will be saved to: {RESULTS_FILE}")
        logger.print(f"Debug log will be saved to: {DEBUG_FILE}")
        input("Press Enter to start tests...")
        
        test_filter_response()
        
        logger.print("\n" + "=" * 80)
        logger.print(f"✓ Test run completed successfully!")
        logger.print(f"Results saved to: {RESULTS_FILE}")
        logger.print(f"Debug log saved to: {DEBUG_FILE}")
        logger.print("=" * 80)
        
    finally:
        logger.close()
