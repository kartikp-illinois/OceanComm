#include "fir_filter.h"
#include <cmath>
#include <algorithm>

FIRFilter::FIRFilter() : delay_line(FILTER_LENGTH, 0.0f), write_index(0), sample_count(0) {
    calculateCoefficients();
}

void FIRFilter::calculateCoefficients() {
    coefficients.resize(FILTER_LENGTH);
    float sum = 0.0f;

    // Temporary vector for centered coefficients (index i -> n = i - N)
    std::vector<float> centered(FILTER_LENGTH);

    for (int i = 0; i < FILTER_LENGTH; i++) {
        int n = i - N; // Convert index to filter coefficient index (-N to N-1)

        double sinc_arg = (B / FS) * (n + 0.5);
        double sinc_val;

        if (std::abs(sinc_arg) < 1e-10) {
            sinc_val = 1.0;
        } else {
            sinc_val = std::sin(M_PI * sinc_arg) / (M_PI * sinc_arg);
        }

        double cos_window = 1.0 + std::cos(M_PI * (n + 0.5) / (N + 0.5));
        centered[i] = static_cast<float>(sinc_val * cos_window);
        sum += centered[i];
    }

    // Normalize centered coefficients (so DC gain = 1)
    if (std::abs(sum) > 1e-6) {
        for (int i = 0; i < FILTER_LENGTH; i++) {
            centered[i] /= sum;
        }
    }

    // Rotate to make coefficients[0] == h[0], coefficients[1] == h[1], ...
    for (int k = 0; k < FILTER_LENGTH; k++) {
        int src = (k + N) % FILTER_LENGTH;
        coefficients[k] = centered[src];
    }
}

float FIRFilter::process(float sample) {
    // Store new sample in circular buffer
    delay_line[write_index] = sample;

    // Standard 32-tap convolution with explicit boundary checking
    float output = 0.0f;
    int index = write_index;

    for (int k = 0; k < FILTER_LENGTH; k++) {
        // Compute the time index: we're accessing x[n-k]
        // where n is the current sample number (sample_count)
        // and we're stepping backwards k samples
        int time_index = sample_count - k;
        
        // Explicitly check if this sample has been received yet
        // If time_index < 0, then x[time_index] = 0 (hasn't arrived, or pre-history)
        float x_sample = 0.0f;
        if (time_index >= 0) {
            // Sample is valid, read from delay line
            x_sample = delay_line[index];
        }
        // else: time_index < 0 means we're accessing x[negative time]
        //       This is explicitly 0, so x_sample stays 0.0f
        
        output += coefficients[k] * x_sample;
        
        // Move to previous sample in circular buffer
        index = (index == 0) ? FILTER_LENGTH - 1 : index - 1;
    }

    // Update for next iteration
    write_index = (write_index + 1) % FILTER_LENGTH;
    sample_count++;

    return output;
}

void FIRFilter::reset() {
    std::fill(delay_line.begin(), delay_line.end(), 0.0f);
    write_index = 0;
    sample_count = 0;
}
