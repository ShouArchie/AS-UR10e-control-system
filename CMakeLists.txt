# CMakeLists.txt for Pico ADC Sampler
cmake_minimum_required(VERSION 3.13)

# Set compiler paths directly
set(CMAKE_C_COMPILER "C:/Program Files/Raspberry Pi/Pico SDK v1.5.1/gcc-arm-none-eabi/bin/arm-none-eabi-gcc.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files/Raspberry Pi/Pico SDK v1.5.1/gcc-arm-none-eabi/bin/arm-none-eabi-g++.exe")
set(CMAKE_ASM_COMPILER "C:/Program Files/Raspberry Pi/Pico SDK v1.5.1/gcc-arm-none-eabi/bin/arm-none-eabi-gcc.exe")

# Set Pico SDK path
set(PICO_SDK_PATH "C:/Program Files/Raspberry Pi/Pico SDK v1.5.1/pico-sdk")

# Include the Pico SDK import script
include(pico_sdk_import.cmake)

# Define the project
project(pico_adc_sampler C CXX ASM)

# Set C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the Pico SDK
pico_sdk_init()

# Add the executable for our ADC sampler
add_executable(pico_adc_sampler
    pico_adc_sampler.cpp
)

# Enable USB serial output
pico_enable_stdio_usb(pico_adc_sampler 1)

# Link required libraries
target_link_libraries(pico_adc_sampler 
    pico_stdlib 
    hardware_adc 
    hardware_dma
)

# Create additional output files (.uf2, .hex, etc.)
pico_add_extra_outputs(pico_adc_sampler)

# Print build success message
message(STATUS "Build configured! Use 'cmake --build build' to compile.") 