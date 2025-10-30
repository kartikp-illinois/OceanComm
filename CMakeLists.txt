cmake_minimum_required(VERSION 3.14)
# Specify the minimum required version of CMake to process this project.
# This ensures compatibility and availability of features used in the build scripts.
# Version 3.14 or above is required here.

project(FIRFilterUDP VERSION 1.0 LANGUAGES CXX)
# Define the project name as FIRFilterUDP.
# Set the project version to 1.0.
# Specify that this project uses C++ (CXX) as the programming language.

set(CMAKE_CXX_STANDARD 14)
# Set the C++ language standard to C++14.
# Ensures your code will be compiled with C++14 features and syntax.

set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enforce that C++14 standard must be used.
# If the compiler does not support C++14, configuration will fail.

add_executable(fir_filter_udp
    src/main.cpp
    src/fir_filter.cpp
)
# Create an executable target named fir_filter_udp.
# The source files compiled into this executable are main.cpp and fir_filter.cpp
# located inside the src directory.

target_include_directories(fir_filter_udp PRIVATE src)
# Specify the include directories for this target.
# PRIVATE means only this target needs these headers (not its dependents).
# Here, the compiler will look for header files in the src folder.

target_link_libraries(fir_filter_udp PRIVATE m)
# Link the math library (libm) to the executable.
# The math library provides standard math functions (like sin, cos).
# PRIVATE means the linkage is only for this target.

target_compile_options(fir_filter_udp PRIVATE -O3 -Wall -Wextra)
# Set compile options for this target:
# -O3 : Apply high optimization level for fastest code generation.
# -Wall : Enable all common compiler warnings.
# -Wextra : Enable extra warning messages for stricter code checks.
# PRIVATE applies only to this target.
