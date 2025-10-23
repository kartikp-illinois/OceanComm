#!/usr/bin/env python3

import socket
import numpy as np
import struct
import time

# Filter parameters
FS = 100e3
B = 10e3  
N = 16
FC_TEST = 1e3   # In passband
FC_TEST2 = 25e3 # Outside passband
PORT = 8080
SERVER_IP = "127.0.0.1"


def calculate_correct_filter():
    """Calculate what the filter coefficients SHOULD be"""
    h_causal = np.zeros(2 * N, dtype=np.float32)
    
    for i in range(2 * N):
        n_val = i - N  # -16 to 15

        sinc_arg = 2 * np.pi * (B / FS) * (n_val + 0.5)
        if abs(sinc_arg) < 1e-10:
            sinc_val = 1.0
        else:
            sinc_val = np.sin(sinc_arg) / sinc_arg  # NO π!
        
        cos_window = 1.0 + np.cos(np.pi * (n_val + 0.5) / (N + 0.5))
        h_causal[i] = sinc_val * cos_window
    
    # Normalize
    h_causal /= np.sum(h_causal)
    return h_causal

def debug_coefficients():
    """Just print what coefficients should be vs what you're getting"""
    expected = calculate_correct_filter()
    print("Expected coefficients (first 16):")
    for i in range(16):
        print(f"  h[{i:2d}] = {expected[i]:.6f}")
    
    # Your current coefficients (from impulse response)
    impulse = np.zeros(50, dtype=np.float32)
    impulse[0] = 1.0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    data = struct.pack(f'{len(impulse)}f', *impulse)
    sock.sendto(data, (SERVER_IP, PORT))
    response, _ = sock.recvfrom(4096)
    received = struct.unpack(f'{len(impulse)}f', response)
    actual = np.array(received[:32])
    
    print("\nActual coefficients (first 16):")
    for i in range(16):
        print(f"  h[{i:2d}] = {actual[i]:.6f}")
    sock.close()


def test_basic_functionality():
    """Simplified test - just check if the basics work"""
    print("=== FIR Filter Basic Test ===")
    
    # Test 1: Simple sine wave
    duration = 0.01  # 10ms
    t = np.arange(0, duration, 1/FS)
    signal = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send entire signal
        data = struct.pack(f'{len(signal)}f', *signal)
        sock.sendto(data, (SERVER_IP, PORT))
        
        # Receive response
        response, addr = sock.recvfrom(4096)
        received = struct.unpack(f'{len(signal)}f', response)
        
        print(f"Sent {len(signal)} samples, received {len(received)} samples")
        
        # Basic checks
        if len(received) != len(signal):
            print(f"❌ FAIL: Length mismatch! Expected {len(signal)}, got {len(received)}")
            return False
        
        # Check if output is reasonable
        input_rms = np.sqrt(np.mean(signal**2))
        output_rms = np.sqrt(np.mean(np.array(received)**2))
        attenuation = 20 * np.log10(output_rms / input_rms)
        
        print(f"Input RMS: {input_rms:.6f}")
        print(f"Output RMS: {output_rms:.6f}") 
        print(f"Attenuation: {attenuation:.2f} dB")
        
        if abs(attenuation) < 3:  # Should pass through
            print("✅ PASS: Signal preserved in passband")
        else:
            print("❌ FAIL: Unexpected attenuation")
            
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    finally:
        sock.close()

def test_impulse_response():
    """Check if filter coefficients match expected"""
    print("\n=== Impulse Response Test ===")
    
    impulse = np.zeros(50, dtype=np.float32)
    impulse[0] = 1.0
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        data = struct.pack(f'{len(impulse)}f', *impulse)
        sock.sendto(data, (SERVER_IP, PORT))
        response, addr = sock.recvfrom(4096)
        received = struct.unpack(f'{len(impulse)}f', response)
        
        # Get actual filter coefficients from impulse response
        actual_coeffs = np.array(received[:32])  # First 32 samples
        
        # Calculate what coefficients SHOULD be
        expected_coeffs = calculate_correct_filter()
        
        print("First 10 expected coefficients:", expected_coeffs[:10])
        print("First 10 actual coefficients:  ", actual_coeffs[:10])
        
        # Check correlation
        correlation = np.corrcoef(expected_coeffs, actual_coeffs)[0,1]
        max_error = np.max(np.abs(expected_coeffs - actual_coeffs))
        
        print(f"Correlation: {correlation:.6f}")
        print(f"Max error: {max_error:.6f}")
        
        if correlation > 0.99:
            print("✅ PASS: Filter coefficients match expected")
        else:
            print("❌ FAIL: Filter coefficients don't match")
            print("This suggests the sinc function implementation is wrong!")
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
    finally:
        sock.close()

def test_packet_continuity():
    """Check if filter state is maintained across packets"""
    print("\n=== Packet Continuity Test ===")
    
    # Generate longer signal
    duration = 0.02
    t = np.arange(0, duration, 1/FS)
    signal = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send in two chunks
        chunk1 = signal[:100]
        chunk2 = signal[100:200]
        
        # Send first chunk
        data1 = struct.pack(f'{len(chunk1)}f', *chunk1)
        sock.sendto(data1, (SERVER_IP, PORT))
        response1, _ = sock.recvfrom(4096)
        out1 = struct.unpack(f'{len(chunk1)}f', response1)
        
        # Send second chunk  
        data2 = struct.pack(f'{len(chunk2)}f', *chunk2)
        sock.sendto(data2, (SERVER_IP, PORT))
        response2, _ = sock.recvfrom(4096)
        out2 = struct.unpack(f'{len(chunk2)}f', response2)
        
        # Check if output is continuous AT THE BOUNDARY (not sample-to-sample)
        boundary_jump = abs(out2[0] - out1[-1])
        
        print(f"Boundary jump between packets: {boundary_jump:.6f}")
        
        # Also check a few samples after boundary to ensure continuity
        post_boundary_consistency = np.mean(np.abs(np.diff(out2[:10])))
        print(f"Post-boundary sample consistency: {post_boundary_consistency:.6f}")
        
        # Reasonable criteria: boundary jump should be small, and filter should stabilize
        if boundary_jump < 0.05 and post_boundary_consistency < 0.1:
            print("✅ PASS: Filter state maintained across packets")
        else:
            print("❌ FAIL: Filter state discontinuity detected")
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
    finally:
        sock.close()

def quick_frequency_test():
    """Quick check of frequency response"""
    print("\n=== Frequency Response Test ===")
    
    for fc in [5e3, 15e3, 25e3]:
        duration = 0.01
        t = np.arange(0, duration, 1/FS)
        signal = np.sin(2 * np.pi * fc * t).astype(np.float32)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        
        try:
            data = struct.pack(f'{len(signal)}f', *signal)
            sock.sendto(data, (SERVER_IP, PORT))
            response, _ = sock.recvfrom(4096)
            received = struct.unpack(f'{len(signal)}f', response)
            
            input_rms = np.sqrt(np.mean(signal**2))
            output_rms = np.sqrt(np.mean(np.array(received)**2))
            attenuation = 20 * np.log10(output_rms / input_rms)
            
            status = "PASS" if (fc <= 10e3 and attenuation > -3) or (fc > 10e3 and attenuation < -10) else "FAIL"
            print(f"{fc/1000:5.0f} kHz: {attenuation:6.1f} dB [{status}]")
            
        except Exception as e:
            print(f"{fc/1000:5.0f} kHz: ERROR - {e}")
        finally:
            sock.close()

def analyze_filter_response():
    """Check what the actual filter frequency response is"""
    print("\n=== Filter Frequency Analysis ===")
    
    # Get filter coefficients from impulse response
    impulse = np.zeros(100, dtype=np.float32)
    impulse[0] = 1.0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    data = struct.pack(f'{len(impulse)}f', *impulse)
    sock.sendto(data, (SERVER_IP, PORT))
    response, _ = sock.recvfrom(4096)
    received = struct.unpack(f'{len(impulse)}f', response)
    h = np.array(received[:32])
    sock.close()
    
    # Compute frequency response
    H = np.fft.fft(h, 1024)
    freq = np.fft.fftfreq(1024, 1/FS)
    
    # Find -3dB point
    magnitude = 20 * np.log10(np.abs(H) + 1e-10)
    max_mag = np.max(magnitude)
    half_power_freq = None
    
    for i in range(len(freq)//2):
        if freq[i] > 0 and magnitude[i] < max_mag - 3:
            half_power_freq = freq[i]
            break
    
    print(f"Filter peak response: {max_mag:.1f} dB")
    print(f"-3dB cutoff frequency: {half_power_freq/1000:.1f} kHz")
    print(f"Expected cutoff: {B/1000:.1f} kHz")
    
    # Plot a few key frequencies
    print("\nKey frequency points:")
    for f in [1e3, 3e3, 5e3, 7e3, 10e3]:
        idx = np.argmin(np.abs(freq - f))
        print(f"  {f/1000:4.0f} kHz: {magnitude[idx]:6.1f} dB")

def debug_continuity_issue():
    """Debug exactly where the continuity problem occurs"""
    print("\n=== Debugging Continuity Issue ===")
    
    # Create a simple rising ramp signal
    signal = np.arange(50, dtype=np.float32) * 0.01  # 0.00, 0.01, 0.02, ...
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send first 25 samples
        chunk1 = signal[:25]
        data1 = struct.pack(f'{len(chunk1)}f', *chunk1)
        sock.sendto(data1, (SERVER_IP, PORT))
        response1, _ = sock.recvfrom(4096)
        out1 = struct.unpack(f'{len(chunk1)}f', response1)
        
        print("First chunk input (last 5):", chunk1[-5:])
        print("First chunk output (last 5):", out1[-5:])
        
        # Send second 25 samples  
        chunk2 = signal[25:]
        data2 = struct.pack(f'{len(chunk2)}f', *chunk2)
        sock.sendto(data2, (SERVER_IP, PORT))
        response2, _ = sock.recvfrom(4096)
        out2 = struct.unpack(f'{len(chunk2)}f', response2)
        
        print("Second chunk input (first 5):", chunk2[:5])
        print("Second chunk output (first 5):", out2[:5])
        
        # Check the boundary
        boundary_jump = abs(out2[0] - out1[-1])
        print(f"Boundary jump: {boundary_jump:.6f}")
        
        # What should happen at the boundary?
        # The filter should maintain state, so output should be continuous
        # If there's a large jump, the delay line is being reset
        
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def debug_large_jumps():
    """Find exactly where the large jumps are happening"""
    print("\n=== Debugging Large Jumps ===")
    
    # Generate test signal
    duration = 0.02
    t = np.arange(0, duration, 1/FS)
    signal = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send in small chunks to find the problem
        chunk_size = 50
        all_output = []
        
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i:i+chunk_size]
            data = struct.pack(f'{len(chunk)}f', *chunk)
            sock.sendto(data, (SERVER_IP, PORT))
            response, _ = sock.recvfrom(4096)
            received = struct.unpack(f'{len(chunk)}f', response)
            all_output.extend(received)
            
            if i > 0:  # Check boundary after first chunk
                prev_chunk_end = all_output[i-1] if i > 0 else 0
                current_chunk_start = received[0]
                jump = abs(current_chunk_start - prev_chunk_end)
                
                if jump > 0.1:
                    print(f"LARGE JUMP at sample {i}: {jump:.6f}")
                    print(f"  Previous output end: {all_output[i-5:i]}")
                    print(f"  Current output start: {received[:5]}")
        
        # Check all sample-to-sample differences
        all_output_array = np.array(all_output)
        diffs = np.diff(all_output_array)
        large_jump_indices = np.where(np.abs(diffs) > 0.1)[0]
        
        if len(large_jump_indices) > 0:
            print(f"\nFound {len(large_jump_indices)} large jumps:")
            for idx in large_jump_indices[:10]:  # Show first 10
                print(f"  Sample {idx}->{idx+1}: jump = {diffs[idx]:.6f}")
        else:
            print("No large jumps found in individual samples")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def test_initial_state():
    """Test if the filter starts with proper initial state"""
    print("\n=== Testing Initial Filter State ===")
    
    # Test 1: Send zeros first to establish baseline
    zeros = np.zeros(100, dtype=np.float32)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        data = struct.pack(f'{len(zeros)}f', *zeros)
        sock.sendto(data, (SERVER_IP, PORT))
        response, _ = sock.recvfrom(4096)
        zero_output = struct.unpack(f'{len(zeros)}f', response)
        
        # Check if filter output settles to zero
        last_10 = zero_output[-10:]
        max_deviation = np.max(np.abs(last_10))
        print(f"Max deviation from zero after 100 zeros: {max_deviation:.6f}")
        
        if max_deviation > 0.01:
            print("❌ Filter doesn't settle to zero properly")
        else:
            print("✅ Filter settles to zero properly")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def analyze_sine_wave_boundary():
    """Analyze why the sine wave has large boundary jumps"""
    print("\n=== Analyzing Sine Wave Boundary Issue ===")
    
    # Generate the exact same signal as in test_packet_continuity()
    duration = 0.02
    t = np.arange(0, duration, 1/FS)
    signal = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    print(f"Signal length: {len(signal)} samples")
    print(f"First chunk (samples 0-99):")
    print(f"  Input range: [{signal[0]:.6f} ... {signal[99]:.6f}]")
    print(f"  Input at boundary: {signal[99]:.6f} -> {signal[100]:.6f}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send first chunk
        chunk1 = signal[:100]
        data1 = struct.pack(f'{len(chunk1)}f', *chunk1)
        sock.sendto(data1, (SERVER_IP, PORT))
        response1, _ = sock.recvfrom(4096)
        out1 = struct.unpack(f'{len(chunk1)}f', response1)
        
        # Send second chunk  
        chunk2 = signal[100:200]
        data2 = struct.pack(f'{len(chunk2)}f', *chunk2)
        sock.sendto(data2, (SERVER_IP, PORT))
        response2, _ = sock.recvfrom(4096)
        out2 = struct.unpack(f'{len(chunk2)}f', response2)
        
        print(f"Output at boundary:")
        print(f"  Chunk1 end:   {out1[-5:]}")
        print(f"  Chunk2 start: {out2[:5]}")
        
        # Calculate what the output SHOULD be if we sent the entire signal
        full_data = struct.pack(f'{len(signal)}f', *signal)
        sock.sendto(full_data, (SERVER_IP, PORT))
        full_response, _ = sock.recvfrom(8192)
        full_output = struct.unpack(f'{len(signal)}f', full_response)
        
        print(f"Full output at boundary:")
        print(f"  Sample 99: {full_output[99]:.6f}")
        print(f"  Sample 100: {full_output[100]:.6f}")
        print(f"  Actual boundary jump in full signal: {abs(full_output[100] - full_output[99]):.6f}")
        
        # Compare chunked vs full
        print(f"Comparison:")
        print(f"  Chunked boundary jump: {abs(out2[0] - out1[-1]):.6f}")
        print(f"  Full signal jump: {abs(full_output[100] - full_output[99]):.6f}")
        print(f"  Expected chunk2[0]: {full_output[100]:.6f}")
        print(f"  Actual chunk2[0]:   {out2[0]:.6f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def check_client_management():
    """Check if the same client filter is being used"""
    print("\n=== Checking Client Management ===")
    
    # Send two packets from the "same client" (same IP/port)
    signal1 = np.sin(2 * np.pi * FC_TEST * np.arange(0, 0.01, 1/FS)).astype(np.float32)
    signal2 = np.sin(2 * np.pi * FC_TEST * np.arange(0.01, 0.02, 1/FS)).astype(np.float32)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # First packet
        data1 = struct.pack(f'{len(signal1)}f', *signal1)
        sock.sendto(data1, (SERVER_IP, PORT))
        response1, addr1 = sock.recvfrom(4096)
        out1 = struct.unpack(f'{len(signal1)}f', response1)
        
        print(f"First packet from: {addr1}")
        
        # Second packet (should be same client)
        data2 = struct.pack(f'{len(signal2)}f', *signal2)
        sock.sendto(data2, (SERVER_IP, PORT))
        response2, addr2 = sock.recvfrom(4096)
        out2 = struct.unpack(f'{len(signal2)}f', response2)
        
        print(f"Second packet from: {addr2}")
        
        if addr1 == addr2:
            print("✅ Same client address - filter state should be maintained")
        else:
            print("❌ DIFFERENT client addresses - this explains the discontinuity!")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def analyze_group_delay_effect():
    """Demonstrate that the issue is group delay, not filter state"""
    print("\n=== Analyzing Group Delay Effect ===")
    
    # Create a signal that crosses zero at the boundary
    duration = 0.02
    t = np.arange(0, duration, 1/FS)
    signal = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    # Find where the sine wave crosses zero
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    print(f"Zero crossings in signal: {zero_crossings[:10]}")
    
    # Check if our boundary (sample 99->100) is near a zero crossing
    boundary_idx = 99
    nearest_zero = zero_crossings[np.argmin(np.abs(zero_crossings - boundary_idx))]
    print(f"Boundary at sample {boundary_idx}, nearest zero crossing at sample {nearest_zero}")
    print(f"Distance to zero crossing: {abs(boundary_idx - nearest_zero)} samples")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send entire signal to see the true output
        full_data = struct.pack(f'{len(signal)}f', *signal)
        sock.sendto(full_data, (SERVER_IP, PORT))
        full_response, _ = sock.recvfrom(8192)
        full_output = struct.unpack(f'{len(signal)}f', full_response)
        
        # The "expected" output at the boundary
        expected_jump = abs(full_output[100] - full_output[99])
        print(f"Expected boundary jump in continuous signal: {expected_jump:.6f}")
        
        # This should be similar to our measured 0.096757
        
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

# Also, let's test with a signal that DOESN'T cross zero at the boundary
def test_non_zero_boundary():
    """Test with a signal that doesn't cross zero at packet boundary"""
    print("\n=== Testing Non-Zero Boundary ===")
    
    # Shift the sine wave so it doesn't cross zero at sample 100
    duration = 0.02
    t = np.arange(0, duration, 1/FS)
    signal = np.sin(2 * np.pi * FC_TEST * t + np.pi/4).astype(np.float32)  # Phase shift
    
    print(f"Input at boundary: {signal[99]:.6f} -> {signal[100]:.6f}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send in two chunks
        chunk1 = signal[:100]
        chunk2 = signal[100:200]
        
        data1 = struct.pack(f'{len(chunk1)}f', *chunk1)
        sock.sendto(data1, (SERVER_IP, PORT))
        response1, _ = sock.recvfrom(4096)
        out1 = struct.unpack(f'{len(chunk1)}f', response1)
        
        data2 = struct.pack(f'{len(chunk2)}f', *chunk2)
        sock.sendto(data2, (SERVER_IP, PORT))
        response2, _ = sock.recvfrom(4096)
        out2 = struct.unpack(f'{len(chunk2)}f', response2)
        
        boundary_jump = abs(out2[0] - out1[-1])
        print(f"Boundary jump with phase-shifted sine: {boundary_jump:.6f}")
        
        if boundary_jump < 0.05:
            print("✅ PASS: Small jump with non-zero boundary")
        else:
            print("❌ Still large - there might be another issue")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def debug_filter_state_directly():
    """Directly test if filter state is being maintained"""
    print("\n=== Direct Filter State Test ===")
    
    # Test 1: Send identical packets and see if output changes
    test_signal = np.ones(50, dtype=np.float32) * 0.5  # Constant signal
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send first packet
        data1 = struct.pack(f'{len(test_signal)}f', *test_signal)
        sock.sendto(data1, (SERVER_IP, PORT))
        response1, _ = sock.recvfrom(4096)
        out1 = struct.unpack(f'{len(test_signal)}f', response1)
        
        print(f"First packet output (last 5): {out1[-5:]}")
        
        # Send second IDENTICAL packet
        data2 = struct.pack(f'{len(test_signal)}f', *test_signal)
        sock.sendto(data2, (SERVER_IP, PORT))
        response2, _ = sock.recvfrom(4096)
        out2 = struct.unpack(f'{len(test_signal)}f', response2)
        
        print(f"Second packet output (first 5): {out2[:5]}")
        
        # For a constant input, the output should stabilize to the same value
        # If filter state is maintained, the second packet should continue smoothly
        
        boundary_jump = abs(out2[0] - out1[-1])
        print(f"Boundary jump with constant input: {boundary_jump:.6f}")
        
        if boundary_jump < 0.01:
            print("✅ Filter state maintained with constant input")
        else:
            print("❌ Filter state NOT maintained - this is the bug!")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def test_circular_buffer():
    """Test if the circular buffer is working correctly"""
    print("\n=== Circular Buffer Test ===")
    
    # Send a pattern that tests the circular buffer wrap-around
    pattern = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    repeated_pattern = np.tile(pattern, 20)  # Repeat 20 times
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        data = struct.pack(f'{len(repeated_pattern)}f', *repeated_pattern)
        sock.sendto(data, (SERVER_IP, PORT))
        response, _ = sock.recvfrom(4096)
        output = struct.unpack(f'{len(repeated_pattern)}f', response)
        
        # The output should show the filter's impulse response repeating
        # If circular buffer is broken, we'll see anomalies
        
        print("Output pattern (first 30 samples):")
        for i in range(0, min(30, len(output)), 5):
            print(f"  Samples {i:2d}-{i+4:2d}: {output[i:i+5]}")
            
        # Check for unexpected jumps
        diffs = np.diff(output)
        large_jumps = np.where(np.abs(diffs) > 0.1)[0]
        if len(large_jumps) > 0:
            print(f"Found {len(large_jumps)} unexpected jumps in pattern test")
        else:
            print("No unexpected jumps in pattern test")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def final_verification():
    """Final test to verify the core functionality"""
    print("\n=== Final Verification ===")
    
    # The key question: Does the filter maintain state across UDP packets?
    # Let's test with a very simple case
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # Send [1, 0, 0, 0, ...] in two chunks
        chunk1 = np.array([1.0] + [0.0] * 49, dtype=np.float32)  # 1 followed by 49 zeros
        chunk2 = np.array([0.0] * 50, dtype=np.float32)  # 50 zeros
        
        # First chunk - should see impulse response
        data1 = struct.pack(f'{len(chunk1)}f', *chunk1)
        sock.sendto(data1, (SERVER_IP, PORT))
        response1, _ = sock.recvfrom(4096)
        out1 = struct.unpack(f'{len(chunk1)}f', response1)
        
        print("First chunk (impulse response start):")
        print(f"  Output: {out1[:10]}")
        
        # Second chunk - should see continuation of impulse response
        data2 = struct.pack(f'{len(chunk2)}f', *chunk2)
        sock.sendto(data2, (SERVER_IP, PORT))
        response2, _ = sock.recvfrom(4096)
        out2 = struct.unpack(f'{len(chunk2)}f', response2)
        
        print("Second chunk (impulse response continuation):")
        print(f"  Output: {out2[:10]}")
        
        # The first output of chunk2 should continue from the impulse response
        expected_continuation = out1[-1]  # Should be close to zero for our filter
        actual_continuation = out2[0]
        
        print(f"Expected continuation: {expected_continuation:.6f}")
        print(f"Actual continuation: {actual_continuation:.6f}")
        
        if abs(actual_continuation - expected_continuation) < 0.01:
            print("✅ IMPULSE RESPONSE CONTINUES - Filter state is maintained!")
        else:
            print("❌ IMPULSE RESPONSE BROKEN - Filter state is NOT maintained!")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        sock.close()

def timed_test(test_func):
    print(f"\nRunning {test_func.__name__}...")
    start_time = time.perf_counter()
    result = test_func()
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"{test_func.__name__} completed in {elapsed_ms:.2f} ms")
    return result

if __name__ == "__main__":
    print("Make sure the C++ server is running on port 8080!")
    print("Press Enter to start tests...")
    input()
    
    timed_test(test_basic_functionality)
    timed_test(test_impulse_response)
    timed_test(test_packet_continuity)
    timed_test(quick_frequency_test)

    test_basic_functionality()
    test_impulse_response()      # This will reveal the sinc function error
    test_packet_continuity()
    quick_frequency_test()
    
    # Add this to your main
    debug_coefficients()
    # Add this call to main
    analyze_filter_response()
    debug_continuity_issue()
    # Add this to your main() after the other tests
    debug_large_jumps()
    test_initial_state()
    analyze_sine_wave_boundary()
    check_client_management()
    analyze_group_delay_effect()
    test_non_zero_boundary()

    debug_filter_state_directly()
    test_circular_buffer() 
    final_verification()
    
    print("\n=== High Level FIR Filter Resource Usage Summary ===")
    print(f"Filter Length (Number of Taps): {2*N}")
    print(f"Per Output Sample Resources:")
    print(f" - Memory Loads from Delay Line: {2*N}")
    print(f" - Memory Loads of Coefficients: {2*N}")
    print(f" - Multiply Operations: {2*N}")
    print(f" - Additions (Accumulator): {2*N - 1}")
    print(f" - Modulo/Bitwise Operations: 1 (write index wrap)")
    print(f"Additional Storage:")
    print(f" - Delay line buffer: {2*N} floats")
    print(f" - Coefficients array: {2*N} floats")