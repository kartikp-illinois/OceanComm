#ifndef FIR_FILTER_H
#define FIR_FILTER_H

#include <vector> // For std::vector
#include <cmath> // For M_PI, std::sin, std::abs

class FIRFilter {
private:
    static constexpr int FILTER_LENGTH = 32; // =total taps
    static constexpr int Filter_Mask = FILTER_LENGTH - 1; //circular buffer indexing
    static constexpr int N = 16;   
    static constexpr double FS = 100e3; //Sampling frequency
    static constexpr double B = 10e3; //Bandwidth

    std::vector<float> coefficients; //stores first half of symmetric coefficients
    std::vector<float> delay_line; //circular buffer for past samples
    int write_index; //current position in delay_line
    long long sample_count;  //total samples processed

    void calculateCoefficients();
    bool isSymmetric() const;

public:
    FIRFilter();

    float process(float sample); //process single sample
    void reset(); //reset filter state

    std::vector<float> getCoefficients() const { //for testing
        std::vector<float> full_coeffs(FILTER_LENGTH);
        for (int i = 0; i < FILTER_LENGTH / 2; i++) {
            full_coeffs[i] = coefficients[i];
            full_coeffs[FILTER_LENGTH - 1 - i] = coefficients[i];
        }
        return full_coeffs;
    }
};

#endif // FIR_FILTER_H
