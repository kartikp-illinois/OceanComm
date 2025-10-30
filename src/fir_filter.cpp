#include "fir_filter.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>


//formula:  h[n] = sinc( (B/FS) * (n - (N - 0.5)) ) * (1 + cos( π * (n - (N - 0.5)) / (N + 0.5) ) )
//as a fourier transform, this is equivalent to a lowpass filter with cutoff frequency B. 
//the cosine window reduces ripples in the frequency response - this is called the windowed-sinc method of designing FIR filters.
//if we did not use the cosine window, the filter would have more ripples in the passband and stopband (Gibbs phenomenon).
// "ripples" are variations in the frequency response that cause some frequencies (especially near the cutoff) to be
        // amplified or attenuated more than others.
//normalizing the filter coefficients would ensure that the sum of the coefficients equals 1.0, which means that
        // the filter does not amplify or attenuate the overall signal level.
        // This is important for preserving the signal's energy and preventing unwanted gain or loss.
        // the energy of the filter is the sum of the squares of its coefficients.
        // it is common practice to normalize the filter coefficients to ensure that the filter has a unity gain at DC (0 Hz).
        // important because it prevents clipping and distortion in audio applications.
        // in radar or sonar applications, it helps maintain the integrity of the received signal.


//this is the constructor for the FIRFilter class. 
//It initializes the delay line with zeros, sets the write index and sample count to zero, 
//and calls the calculateCoefficients() method to compute the filter coefficients.
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

    //calc all coefficients by passing impulse through sinc function and cosine window
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
