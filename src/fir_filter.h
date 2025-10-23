#ifndef FIR_FILTER_H
#define FIR_FILTER_H

#include <vector>
#include <cmath>

class FIRFilter {
private:
    static constexpr int FILTER_LENGTH = 32; // =total taps
    static constexpr int Filter_Mask = FILTER_LENGTH - 1; //circular buffer indexing
    static constexpr int N = 16;   
    static constexpr double FS = 100e3; //Sampling frequency
    static constexpr double B = 10e3; //Bandwidth

    std::vector<float> coefficients;
    std::vector<float> delay_line;
    int write_index;
    long long sample_count;  //total samples processed

    void calculateCoefficients();
    bool isSymmetric() const;

public:
    FIRFilter();

    float process(float sample);
    void reset();

    std::vector<float> getCoefficients() const { 
        std::vector<float> full_coeffs(FILTER_LENGTH);
        for (int i = 0; i < FILTER_LENGTH / 2; i++) {
            full_coeffs[i] = coefficients[i];
            full_coeffs[FILTER_LENGTH - 1 - i] = coefficients[i];
        }
        return full_coeffs;
    }
};

#endif // FIR_FILTER_H
