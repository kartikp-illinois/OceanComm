#ifndef FIR_FILTER_H
#define FIR_FILTER_H

#include <vector>

class FIRFilter {
public:
    FIRFilter();
    float process(float sample);
    void reset();
};

#endif // FIR_FILTER_H
