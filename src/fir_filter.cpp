#include "fir_filter.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

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
    coefficients.resize(FILTER_LENGTH / 2);
    
    std::vector<float> full_coeffs(FILTER_LENGTH);
    float sum = 0.0f;

    //calc all coefficients
    for (int k = 0; k < FILTER_LENGTH; k++) {
        int n = k - N;
        double sinc_arg = (B / FS) * (n + 0.5);
        double sinc_val;
        
        if (std::abs(sinc_arg) < 1e-10) {
            sinc_val = 1.0;
        } else {
            sinc_val = std::sin(sinc_arg) / sinc_arg;
        }
        
        double cos_window = 1.0 + std::cos(M_PI * (n + 0.5) / (N + 0.5));
        full_coeffs[k] = static_cast<float>(sinc_val * cos_window);
        // sum += full_coeffs[k];
    }


    // // Normalize
    // if (std::abs(sum) > 1e-6) {
    //     for (int i = 0; i < FILTER_LENGTH; i++) {
    //         full_coeffs[i] /= sum;
    //     }
    // }

    // Store only first half
    for (int i = 0; i < FILTER_LENGTH / 2; i++) {
        coefficients[i] = full_coeffs[i];
    }
}

float FIRFilter::process(float sample) {
    // Write new sample
    delay_line[write_index] = sample;
    
    float output = 0.0f;
    
    //h[i] = h[FILTER_LENGTH - 1 - i]
    //even-length symmetric filter: y[n] = h[0]*(x[n] + x[n-31]) + h[1]*(x[n-1] + x[n-30]) + ... + h[15]*(x[n-15] + x[n-16])
    
    int half_length = FILTER_LENGTH / 2;
    
    for (int i = 0; i < half_length; i++) {
        //newer sample position
        int pos_new = (write_index - i + FILTER_LENGTH) & Filter_Mask;
        
        //older sample position
        int pos_old = (write_index - (FILTER_LENGTH - 1 - i) + FILTER_LENGTH) & Filter_Mask;
        
        float sample_new = delay_line[pos_new];
        float sample_old = delay_line[pos_old];
        
        float sum = sample_new + sample_old;
        float product = coefficients[i] * sum;
        output += product;
    }
    
    // if (sample_count < 5) {
    //     std::cout << "[C++]   Output: " << std::setprecision(10) << output << "\n";
    // }

    //Move write pointer forward for next sample
    write_index = (write_index + 1) & Filter_Mask;
    sample_count++;
    
    return output;
}

void FIRFilter::reset() {
    std::fill(delay_line.begin(), delay_line.end(), 0.0f);
    write_index = 0;
    sample_count = 0;
}
