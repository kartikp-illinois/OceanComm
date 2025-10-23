#ifndef FIR_FILTER_H
#define FIR_FILTER_H

#include <vector>
#include <cmath>

class FIRFilter {
private:
    static constexpr int FILTER_LENGTH = 32; // total taps
    static constexpr int Filter_Mask = FILTER_LENGTH - 1; // for circular buffer indexing
    static constexpr int N = 16;             // half the filter length
    static constexpr double FS = 100e3;      // Sampling frequency
    static constexpr double B = 10e3;        // Bandwidth

    std::vector<float> coefficients;
    std::vector<float> delay_line;
    int write_index;
    long long sample_count;  // Track total samples received (for explicit boundary checking)

    void calculateCoefficients();

public:
    FIRFilter();

    float process(float sample);
    void reset();

    const std::vector<float>& getCoefficients() const { return coefficients; }
};

#endif // FIR_FILTER_H
