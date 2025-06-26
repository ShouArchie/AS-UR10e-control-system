/**
 * High-performance 40kHz ADC sampler for Raspberry Pi Pico.
 * 
 * - Uses hardware timer interrupt to trigger ADC conversions at precise 40kHz.
 * - Sends data in batches like the working MicroPython version.
 * - Maintains compatibility with the existing Python grapher.
 */

#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/timer.h"

// --- Configuration ---
#define ADC_PIN_NUM 26      // GPIO 26 is ADC 0
#define ADC_CHANNEL 0       // ADC channel for GPIO 26
#define SAMPLE_RATE 40000   // Target sample rate in Hz
#define BATCH_SIZE 400      // Send 400 samples per batch (10ms of data at 40kHz)

// --- Globals ---
uint16_t sample_buffer[BATCH_SIZE];
volatile int buffer_index = 0;
volatile bool buffer_ready = false;

// Timer interrupt callback - this must be fast!
bool timer_callback(struct repeating_timer *t) {
    if (!buffer_ready) {
        // Read ADC - use the exact same scaling as MicroPython's read_u16()
        uint16_t raw_adc = adc_read();
        // MicroPython scales 12-bit (0-4095) to 16-bit (0-65535) using bit shifting
        // This matches the MicroPython implementation more closely
        sample_buffer[buffer_index] = raw_adc << 4;  // Left shift by 4 bits (multiply by 16)
        
        buffer_index++;
        if (buffer_index >= BATCH_SIZE) {
            buffer_ready = true;
            buffer_index = 0;
        }
    }
    return true; // Continue repeating
}

int main() {
    // Initialize standard I/O for USB serial communication
    stdio_init_all();

    // Wait for USB serial to be ready
    sleep_ms(2000);
    
    printf("Pi Pico 40kHz C++ ADC Reader Started\n");
    printf("Sampling GPIO%d at %d Hz.\n", ADC_PIN_NUM, SAMPLE_RATE);
    printf("Sending batches of %d samples.\n", BATCH_SIZE);

    // --- ADC Setup ---
    adc_init();
    adc_gpio_init(ADC_PIN_NUM);
    adc_select_input(ADC_CHANNEL);
    
    // Use default ADC clock (48MHz) for maximum sensitivity
    // Slower clocks reduce noise but might also reduce sensitivity to fast transients

    // Set up the timer interrupt for precise sampling
    struct repeating_timer timer;
    int64_t timer_period_us = 1000000 / SAMPLE_RATE;  // Period in microseconds
    
    // Start the timer interrupt
    add_repeating_timer_us(-timer_period_us, timer_callback, NULL, &timer);
    
    printf("READY\n");

    while (1) {
        if (buffer_ready) {
            // Send the batch header and data
            printf("BATCH,%d\n", BATCH_SIZE);
            for (int i = 0; i < BATCH_SIZE; i++) {
                printf("%u\n", sample_buffer[i]);
            }
            printf("END_BATCH\n");
            
            // Mark buffer as processed
            buffer_ready = false;
        }
        
        // Small delay to prevent busy waiting
        sleep_ms(1);
    }

    return 0;
} 