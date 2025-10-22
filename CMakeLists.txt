cmake_minimum_required(VERSION 3.14)
project(FIRFilterUDP VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fir_filter_udp
    src/main.cpp
    src/fir_filter.cpp
)

target_include_directories(fir_filter_udp PRIVATE src)
target_link_libraries(fir_filter_udp PRIVATE m)
target_compile_options(fir_filter_udp PRIVATE -O3 -Wall -Wextra)
