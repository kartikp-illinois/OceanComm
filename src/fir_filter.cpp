#include "fir_filter.h" // Our FIR filter implementation
#include <cmath> // For M_PI, std::sin, std::abs
#include <algorithm> // For std::fill, std::max
#include <iostream> // For debug output
#include <iomanip> // For std::setprecision

FIRFilter::FIRFilter() : delay_line(FILTER_LENGTH, 0.0f), write_index(0), sample_count(0) {
    calculateCoefficients();
}

bool FIRFilter::isSymmetric() const { //for test - make sure h is actually symmetric
    int half = FILTER_LENGTH / 2;
    for (int i = 0; i < half; i++) {
        if (std::abs(coefficients[i] - coefficients[FILTER_LENGTH - 1 - i]) > 1e-6) {
            return false;
        }
    }
    return true;
}


void FIRFilter::calculateCoefficients() {
    // Resize coefficient vector to store half the filter length due to symmetry
    coefficients.resize(FILTER_LENGTH / 2);
    
    // Temporary full coefficient array (length = FILTER_LENGTH)
    std::vector<float> full_coeffs(FILTER_LENGTH);

    float sum = 0.0f; // Accumulate sum of coefficients for normalization

    // std::cout << "\n[C++] calculateCoefficients() called\n";
    // std::cout << "[C++] FILTER_LENGTH = " << FILTER_LENGTH << ", N = " << N << "\n";
    // std::cout << "[C++] Storing only first half due to symmetry\n";

    // Loop over each filter tap index k = 0...31 (for FILTER_LENGTH=32)
    // a tap is essentially a point along the delay line where a sample is "tapped" and sent to a multiplier.
    for (int k = 0; k < FILTER_LENGTH; k++) {
        // Convert 0-based index k to symmetric index n centered at zero:
        // n will go from -N to N-1, e.g. -16 to 15 for N=16
        int n = k - N;

        // Compute argument of sinc function (angular frequency form):
        // The raw formula for ideal sinc lowpass is:
        // h[n] = sinc(2 * Fc / Fs * (n + 0.5)) * window
        //
        // Here, 2 * M_PI * (B/FS) * (n + 0.5)
        //
        // Explanation of terms:
        // M_PI: mathematical π, ~3.14159
        // B: cutoff bandwidth (Fc)
        // FS: sampling frequency
        // (n + 0.5): shift to center the FIR filter properly
        //
        // Multiplying by 2π converts cycles/second (Hz) to radians/second (angular frequency)
        // Angular frequency ω = 2πf
        //
        // This scaling makes the sinc formula compatible with sin(x)/x where x is in radians.
        // However, the 2 pi is completely wrong. if we were to use the proper normalized sinc definiton,
        // we should have used pi. this was an oversight, which stems from the angular frequency thing i was thinkign abt.
        double sinc_arg = 2 * M_PI * (B / FS) * (n + 0.5);

        double sinc_val;
        
        // Handle the removable singularity at sinc(0), where sin(0)/0 is undefined mathematically
        // By limit, sinc(0) = 1, so check if sinc_arg near zero to avoid division by zero error
        if (std::abs(sinc_arg) < 1e-10) {
            sinc_val = 1.0;
        } else {
            // Calculate normalized sinc value sin(x)/x for filter impulse response
            sinc_val = std::sin(sinc_arg) / sinc_arg;
        }
        
        // Calculate the cosine window value at this sample index n
        // The window reduces ripples in frequency response (Gibbs phenomenon)
        // Formula: w[n] = 1 + cos(π * (n + 0.5) / (N + 0.5))
        // Gives a raised cosine shape centered over the filter length
        double cos_window = 1.0 + std::cos(M_PI * (n + 0.5) / (N + 0.5));

        // Calculate the raw coefficient value combining sinc and window
        full_coeffs[k] = static_cast<float>(sinc_val * cos_window);

        // Accumulate sum for normalization later
        sum += full_coeffs[k];
    }


    // Normalize the coefficients so sum(h[n]) = 1 (to preserve signal magnitude)
    if (std::abs(sum) > 1e-6) {
        for (int i = 0; i < FILTER_LENGTH; i++) {
            full_coeffs[i] /= sum;
        }
    }

    // Store only first half
    for (int i = 0; i < FILTER_LENGTH / 2; i++) {
        coefficients[i] = full_coeffs[i];
    }


    // float check_sum = 0.0f;
    // for (int i = 0; i < FILTER_LENGTH; i++) {
    //     check_sum += full_coeffs[i];
    // }
    // std::cout << "[C++] Sum after normalization: " << std::setprecision(10) 
    //           << check_sum << "\n";
}

float FIRFilter::process(float sample) {
    // Write new sample
    delay_line[write_index] = sample;
    
    float output = 0.0f;
    
    // // DEBUG: Show first few samples
    // if (sample_count < 5) {
    //     std::cout << "\n[C++] process() call #" << sample_count << "\n";
    //     std::cout << "[C++]   Input sample: " << std::setprecision(10) << sample << "\n";
    //     std::cout << "[C++]   write_index: " << write_index << "\n";
    //     std::cout << "[C++]   Using symmetric optimization\n";
    // }
    
    // symmetry: h[i] = h[FILTER_LENGTH - 1 - i]
    // For even-length filter:
    // y[n] = h[0]*(x[n] + x[n-31]) + h[1]*(x[n-1] + x[n-30]) + ... + h[15]*(x[n-15] + x[n-16])
    
    int half_length = FILTER_LENGTH / 2;
    
    for (int i = 0; i < half_length; i++) {
        // Position of newer sample (going backwards from write_index)
        int pos_new = (write_index - i + FILTER_LENGTH) & Filter_Mask;
        
        // Position of older sample (symmetric counterpart)
        int pos_old = (write_index - (FILTER_LENGTH - 1 - i) + FILTER_LENGTH) & Filter_Mask;
        
        // Get both samples
        float sample_new = delay_line[pos_new];
        float sample_old = delay_line[pos_old];
        
        // Add them first, then multiply by coefficient (exploit symmetry)
        float sum = sample_new + sample_old;
        float product = coefficients[i] * sum;
        output += product;
        
        // // DEBUG: Show first few multiplications
        // if (sample_count < 5 && i < 3) {
        //     std::cout << "[C++]   i=" << i 
        //               << " pos_new=" << pos_new << " pos_old=" << pos_old
        //               << " sample_new=" << std::setprecision(10) << sample_new
        //               << " sample_old=" << std::setprecision(10) << sample_old
        //               << " sum=" << std::setprecision(10) << sum
        //               << " coeff=" << std::setprecision(10) << coefficients[i]
        //               << " product=" << std::setprecision(10) << product << "\n";
        // }
    }
    
    // if (sample_count < 5) {
    //     std::cout << "[C++]   Output: " << std::setprecision(10) << output << "\n";
    //     std::cout << "[C++]   Multiplications used: " << half_length 
    //               << " (vs " << FILTER_LENGTH << " without symmetry)\n";
    // }
    
    // Move write pointer forward for next sample
    write_index = (write_index + 1) & Filter_Mask;
    sample_count++;
    
    return output;
}

void FIRFilter::reset() {
    std::fill(delay_line.begin(), delay_line.end(), 0.0f);
    write_index = 0;
    sample_count = 0;
}
