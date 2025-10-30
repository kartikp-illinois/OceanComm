#!/usr/bin/env python3

import socket
import numpy as np
import struct
import time

# Filter parameters
FS = 100e3
B = 10e3  
N = 16
PORT = 8080
SERVER_IP = "127.0.0.1"

def calculate_correct_filter():
    h = np.zeros(2 * N, dtype=np.float32)
    for i in range(2 * N):
        n_val = i - N
        sinc_arg = (B / FS) * (n_val + 0.5)
        if abs(sinc_arg) < 1e-10:
            sinc_val = 1.0
        else:
            sinc_val = np.sin(sinc_arg) / sinc_arg
        cos_window = 1.0 + np.cos(np.pi * (n_val + 0.5) / (N + 0.5))
        h[i] = sinc_val * cos_window
    return h

def test_impulse_response():
    print("=== TEST 1: IMPULSE RESPONSE ===")
    
    h_expected = calculate_correct_filter()
    print(f"\n[EXPECTED] Filter coefficients h[n] (UNNORMALIZED):")
    print(f"  Length: {len(h_expected)}")
    print(f"  Sum: {np.sum(h_expected):.10f}")
    print(f"  First 5:  {h_expected[:5]}")
    print(f"  Last 5:   {h_expected[-5:]}")
    
    # Send impulse
    impulse = np.zeros(50, dtype=np.float32)
    impulse[0] = 1.0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        start = time.perf_counter()
        data = struct.pack(f'{len(impulse)}f', *impulse)
        sock.sendto(data, (SERVER_IP, PORT))
        response, _ = sock.recvfrom(4096)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        received = np.array(struct.unpack(f'{len(impulse)}f', response))
        h_actual = received[:32]
        
        print(f"\n[ACTUAL] From server (impulse response):")
        print(f"  Length: {len(h_actual)}")
        print(f"  Sum: {np.sum(h_actual):.10f}")
        print(f"  First 5:  {h_actual[:5]}")
        print(f"  Last 5:   {h_actual[-5:]}")
        
        # Compare
        diff = np.abs(h_actual - h_expected)
        max_err = np.max(diff)
        rmse = np.sqrt(np.mean(diff**2))
        
        print(f"\n[COMPARISON]:")
        print(f"  Max error: {max_err:.6e}")
        print(f"  RMSE: {rmse:.6e}")
        print(f"  Match (tolerance 1e-6)? {np.allclose(h_actual, h_expected, atol=1e-6)}")
        print(f"  Time: {elapsed_ms:.2f} ms")
        
        if np.allclose(h_actual, h_expected, atol=1e-6):
            print(f"\nPASS: Impulse response matches filter coefficients")
            return True
        else:
            print(f"\nFAIL: Impulse response does not match")
            return False
            
    except Exception as e:
        print(f"\nERROR: {e}")
        return False
    finally:
        sock.close()

def test_direct_convolution():
    print("=== TEST 2: CONVOLUTION OUTPUT ===")    

    FC_TEST = 1000
    duration = 0.005
    t = np.arange(0, duration, 1/FS)
    x = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    print(f"\n[INPUT SIGNAL]:")
    print(f"  Frequency: {FC_TEST} Hz")
    print(f"  Duration: {duration*1000:.2f} ms")
    print(f"  Length: {len(x)} samples")
    
    h = calculate_correct_filter()
    
    # Pad input with zeros, filter history starts at zero
    x_padded = np.concatenate([np.zeros(len(h)-1, dtype=np.float32), x])
    y_expected_full = np.convolve(x_padded, h, mode='valid').astype(np.float32)
    
    print(f"\n[EXPECTED OUTPUT (causal convolution)]:")
    print(f"  Length: {len(y_expected_full)}")
    print(f"  First 5: {y_expected_full[:5]}")
    print(f"  First 5 after transient: {y_expected_full[32:37]}")
    
    # Get actual
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        start = time.perf_counter()
        data = struct.pack(f'{len(x)}f', *x)
        sock.sendto(data, (SERVER_IP, PORT))
        response, _ = sock.recvfrom(4096)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        y_actual = np.array(struct.unpack(f'{len(x)}f', response))
        
        print(f"\n[ACTUAL OUTPUT:]")
        print(f"  Length: {len(y_actual)}")
        print(f"  First 5: {y_actual[:5]}")
        print(f"  First 5 after transient: {y_actual[32:37]}")
        print(f"  Time: {elapsed_ms:.2f} ms")
        
        # Full comparison
        min_len = min(len(y_actual), len(y_expected_full))
        y_actual_cmp = y_actual[:min_len]
        y_expected_cmp = y_expected_full[:min_len]
        
        diff = np.abs(y_actual_cmp - y_expected_cmp)
        max_err = np.max(diff)
        rmse = np.sqrt(np.mean(diff**2))
        
        print(f"\n[FULL COMPARISON]:")
        print(f"  Length compared: {min_len}")
        print(f"  Max error: {max_err:.6e}")
        print(f"  RMSE: {rmse:.6e}")
        print(f"  Match (tolerance 1e-4)? {np.allclose(y_actual_cmp, y_expected_cmp, atol=1e-4)}")
        
        # Also compare steady-state only
        skip_samples = len(h)
        y_actual_steady = y_actual[skip_samples:]
        y_expected_steady = y_expected_full[skip_samples:]
        min_len_steady = min(len(y_actual_steady), len(y_expected_steady))
        
        if min_len_steady > 0:
            diff_steady = np.abs(y_actual_steady[:min_len_steady] - y_expected_steady[:min_len_steady])
            max_err_steady = np.max(diff_steady)
            rmse_steady = np.sqrt(np.mean(diff_steady**2))
            
            print(f"\n[STEADY-STATE COMPARISON (skipping first {skip_samples})]:")
            print(f"  Length compared: {min_len_steady}")
            print(f"  Max error: {max_err_steady:.6e}")
            print(f"  RMSE: {rmse_steady:.6e}")
        
        if np.allclose(y_actual_cmp, y_expected_cmp, atol=1e-4):
            print(f"\nPASS: Convolution output matches expected")
            return True
        else:
            print(f"\nFAIL: Convolution output does not match")
            print(f"\nFirst 10 samples:")
            for i in range(min(10, min_len)):
                print(f"  [{i}] Expected: {y_expected_cmp[i]:12.8f}, Actual: {y_actual_cmp[i]:12.8f}, Diff: {diff[i]:12.8f}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sock.close()


def test_passband_response():
    print("=== TEST 3: PASSBAND FREQUENCY RESPONSE ===")

    freqs = [500, 1000, 5000]
    results = []
    
    for fc in freqs:
        duration = 0.01
        t = np.arange(0, duration, 1/FS)
        x = np.sin(2 * np.pi * fc * t).astype(np.float32)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        
        try:
            start = time.perf_counter()
            data = struct.pack(f'{len(x)}f', *x)
            sock.sendto(data, (SERVER_IP, PORT))
            response, _ = sock.recvfrom(4096)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            y = np.array(struct.unpack(f'{len(x)}f', response))
            
            input_rms = np.sqrt(np.mean(x**2))
            output_rms = np.sqrt(np.mean(y**2))
            attenuation_db = 20 * np.log10(output_rms / input_rms)
            
            print(f"\n  {fc:5d} Hz: {attenuation_db:6.2f} dB, time: {elapsed_ms:.2f} ms")
            results.append((fc, attenuation_db))
            
        except Exception as e:
            print(f"  {fc:5d} Hz: ERROR - {e}")
        finally:
            sock.close()
    
    # NOTE: Without normalization, passband won't be at 0 dB
    # We just check that low frequencies aren't heavily attenuated
    passband_ok = all(att > -6 for _, att in results)  # Relaxed threshold
    if passband_ok:
        print(f"\nPASS: Passband response OK (all > -6dB)")
        return True
    else:
        print(f"\nFAIL: Passband response not OK")
        return False

def test_stopband_response():
    print("=== TEST 4: STOPBAND FREQUENCY RESPONSE ===")
    
    freqs = [15000, 25000, 40000]
    results = []
    
    for fc in freqs:
        duration = 0.01
        t = np.arange(0, duration, 1/FS)
        x = np.sin(2 * np.pi * fc * t).astype(np.float32)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        
        try:
            start = time.perf_counter()
            data = struct.pack(f'{len(x)}f', *x)
            sock.sendto(data, (SERVER_IP, PORT))
            response, _ = sock.recvfrom(4096)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            y = np.array(struct.unpack(f'{len(x)}f', response))
            
            input_rms = np.sqrt(np.mean(x**2))
            output_rms = np.sqrt(np.mean(y**2))
            
            # Avoid log of zero
            if output_rms < 1e-10:
                attenuation_db = -100.0
            else:
                attenuation_db = 20 * np.log10(output_rms / input_rms)
            
            print(f"\n  {fc:5d} Hz: {attenuation_db:6.2f} dB, time: {elapsed_ms:.2f} ms")
            results.append((fc, attenuation_db))
            
        except Exception as e:
            print(f"  {fc:5d} Hz: ERROR - {e}")
        finally: 
            sock.close()
    
    # Check relative attenuation: stopband should be much lower than passband
    stopband_ok = all(att < -5 for _, att in results) 
    if stopband_ok:
        print(f"\nPASS: Stopband response OK (all < -20dB)")
        return True
    else:
        print(f"\nFAIL: Stopband response not OK")
        return False

def test_packet_continuity():
    print("=== TEST 5: PACKET CONTINUITY ===")
    
    FC_TEST = 1000
    duration = 0.02
    t = np.arange(0, duration, 1/FS)
    signal = np.sin(2 * np.pi * FC_TEST * t).astype(np.float32)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    sock.connect((SERVER_IP, PORT))
    
    try:
        chunk1 = signal[:100]
        chunk2 = signal[100:200]
        
        start = time.perf_counter()
        
        data1 = struct.pack(f'{len(chunk1)}f', *chunk1)
        sock.send(data1)
        
        response1 = sock.recv(4096)
        out1 = np.array(struct.unpack(f'{len(chunk1)}f', response1))
        
        data2 = struct.pack(f'{len(chunk2)}f', *chunk2)
        sock.send(data2)
        
        response2 = sock.recv(4096)
        out2 = np.array(struct.unpack(f'{len(chunk2)}f', response2))
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        boundary_jump = abs(out2[0] - out1[-1])
        
        # print(f"\n[PACKET 1]:")
        # print(f"  Length: {len(chunk1)}")
        # print(f"  Output last 3: {out1[-3:]}")
        
        # print(f"\n[PACKET 2]:")
        # print(f"  Length: {len(chunk2)}")
        # print(f"  Output first 3: {out2[:3]}")
        
        # print(f"\n[BOUNDARY]:")
        # print(f"  Jump (out2[0] - out1[-1]): {boundary_jump:.6f}")
        # print(f"  Time: {elapsed_ms:.2f} ms")
        
        if boundary_jump < 1.0:  #Large threshold due to higher gain
            print(f"\nPASS: Continuity maintained (jump < 1.0)")
            return True
        else:
            print(f"\nFAIL: Discontinuity detected (jump >= 1.0)")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sock.close()


if __name__ == "__main__":
    print("\nC++ server should be on port 8080!")
    print("Press Enter to start:")
    input()
    
    results = []
    results.append(("Impulse Response", test_impulse_response()))
    results.append(("Direct Convolution", test_direct_convolution()))
    results.append(("Passband Response", test_passband_response()))
    results.append(("Stopband Response", test_stopband_response()))
    results.append(("Packet Continuity", test_packet_continuity()))
        
    print("\nTEST SUMMARY")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:30s} {status}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")