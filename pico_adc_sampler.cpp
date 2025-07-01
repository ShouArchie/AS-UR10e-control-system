#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/adc.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/clocks.h"
#include "pico/stdio.h"

#define ADC_PIN_NUM 26      // GPIO 26 is ADC 0
#define ADC_CHANNEL 0       // ADC channel for GPIO 26
#define SAMPLE_RATE 60000   // 60kHz sampling rate
#define SAMPLES_PER_BATCH 3000    // 0.05 second worth of samples at 60kHz

volatile uint32_t sample_index = 0;
static uint16_t raw_buffer[SAMPLES_PER_BATCH];          // Raw 12-bit ADC samples
static volatile uint32_t batch_number = 0;

// Pico ADC reference voltage (used only for host-side conversion)
#define ADC_REF_VOLTAGE 3.3f
#define ADC_MAX_VALUE 4095.0f

static inline void send_bytes(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        putchar_raw((int)data[i]);
    }
}

// Binary packet format: 0xA5 0x5A | uint32_t batch_id | uint16_t samples[3000]
void send_batch_binary() {
    const uint8_t header[2] = { 0xA5, 0x5A };
    send_bytes(header, 2);
    send_bytes((const uint8_t*)&batch_number, sizeof(batch_number));
    send_bytes((const uint8_t*)raw_buffer, sizeof(uint16_t) * SAMPLES_PER_BATCH);
}

int main() {
    // Initialise USB stdio (includes CDC)
    stdio_init_all();

    // Wait until host opens the port – avoids losing the banner
    while (!stdio_usb_connected()) {
        sleep_ms(100);
    }
    
    printf("=== PICO FAST SAMPLER v4.0 ===\n");
    printf("Config: 60kHz, 3k samples per batch, 0.05s intervals\n");
    printf("GPIO: 26 (ADC0)\n");
    printf("Optimized for speed\n");
    fflush(stdout);
    
    // ---------- ADC CONFIGURATION ----------
    adc_init();
    adc_gpio_init(ADC_PIN_NUM);
    adc_select_input(ADC_CHANNEL);

    // Set ADC clock so that sampling rate is exactly SAMPLE_RATE
    // ADC clock source is 48 MHz; divisor = 48 000 000 / SAMPLE_RATE
    adc_set_clkdiv(48'000'000.0f / SAMPLE_RATE);

    // Configure FIFO: enable, DREQ on 1 sample, no ERR bit, no shift
    adc_fifo_setup(true, true, 1, false, false);
    adc_run(true);

    // ---------- DMA CONFIGURATION ----------
    int dma_chan = dma_claim_unused_channel(true);
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_16);
    channel_config_set_read_increment(&cfg, false);   // Always read from ADC FIFO
    channel_config_set_write_increment(&cfg, true);   // Increment destination buffer
    channel_config_set_dreq(&cfg, DREQ_ADC);          // Pace on ADC ready

    printf("Status: Ready for fast sampling...\n");
    fflush(stdout);
    
    // ------------- MAIN ACQUISITION LOOP (DMA) -------------
    while (true) {
        // Kick DMA – it will block until SAMPLES_PER_BATCH samples are in raw_buffer
        dma_channel_configure(
            dma_chan,
            &cfg,
            raw_buffer,           // dst
            &adc_hw->fifo,        // src
            SAMPLES_PER_BATCH,    // length
            true                  // start immediately and BLOCK (we wait for finish)
        );

        // Send the batch in binary (raw ADC counts)
        send_batch_binary();
        batch_number++;
    }
    
    return 0;
} 