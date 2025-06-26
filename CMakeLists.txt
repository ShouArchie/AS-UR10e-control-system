# This is the build configuration file for CMake.

# Boilerplate: Set minimum required version of CMake.
cmake_minimum_required(VERSION 3.13)

# This pulls in the Pico SDK, which is required.
# It will either find it from the PICO_SDK_PATH environment variable
# or from a git submodule.
include(pico_sdk_import.cmake)

# Define the project name and specify it's a C/C++ project.
project(adc_sampler_40khz C CXX ASM)

# Set the C standard to C11 and C++ standard to C++17.
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the Pico SDK for our project.
pico_sdk_init()

# Add our source file to create an executable.
# The name of the final executable will be "adc_sampler".
add_executable(adc_sampler
    adc_sampler.cpp
)

# Enable USB serial output for our executable.
pico_enable_stdio_usb(adc_sampler 1)

# Link the required hardware libraries to our executable.
# - hardware_adc is needed for ADC functions.
# - hardware_timer is needed for timer interrupt functions.
target_link_libraries(adc_sampler pico_stdlib hardware_adc hardware_timer)

# This command creates the final .uf2 file that you can drag and drop
# onto the Pico's bootloader drive.
pico_add_extra_outputs(adc_sampler)

# Print a message after the build is done.
message(STATUS "Build successful! Drag and drop adc_sampler.uf2 onto your Pico.") 