#include "fir_filter.h"
#include <cmath>
#include <algorithm>

FIRFilter::FIRFilter() : delay_line(FILTER_LENGTH, 0.0f), write_index(0), sample_count(0) {
    calculateCoefficients();
}

void FIRFilter::calculateCoefficients() {
    coefficients.resize(FILTER_LENGTH);
    float sum = 0.0f;

    // Calculate coefficients directly in natural order
    for (int k = 0; k < FILTER_LENGTH; k++) {
        int n = k - N;  // This gives the correct time index

        double sinc_arg = 2 * M_PI * (B / FS) * (n + 0.5);
        double sinc_val;
        
        if (std::abs(sinc_arg) < 1e-10) {
            sinc_val = 1.0;
        } else {
            sinc_val = std::sin(sinc_arg) / sinc_arg;
        }
        
        double cos_window = 1.0 + std::cos(M_PI * (n + 0.5) / (N + 0.5));
        coefficients[k] = static_cast<float>(sinc_val * cos_window);
        sum += coefficients[k];
    }

    // Normalize
    if (std::abs(sum) > 1e-6) {
        for (int i = 0; i < FILTER_LENGTH; i++) {
            coefficients[i] /= sum;
        }
    }
}

float FIRFilter::process(float sample) {
    // Update delay line - new sample at current position
    delay_line[write_index] = sample;
    
    // Perform convolution
    float output = 0.0f;
    int index = write_index;
    
    for (int i = 0; i < FILTER_LENGTH; i++) {
        output += coefficients[i] * delay_line[index];
        // Move backwards through delay line
        index = (index == 0) ? Filter_Mask : index - 1;
    }
    
    // Advance write position
    write_index = (write_index + 1) & Filter_Mask;
    
    return output;
}

void FIRFilter::reset() {
    std::fill(delay_line.begin(), delay_line.end(), 0.0f);
    write_index = 0;
    sample_count = 0;
}
