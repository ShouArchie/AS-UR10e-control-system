# Pi Pico 5kHz ADC Reader - FIXED VERSION
# Addresses Thonny shell issues and data corruption
# Use this instead of the original pico_adc_reader.py

import machine
import sys
import time

# --- Configuration ---
ADC_PIN = 26
SAMPLE_RATE_HZ = 5000
BATCH_SIZE = 250  # Send 250 samples at a time (50ms of data)

# --- Globals ---
adc = machine.ADC(ADC_PIN)
data_buffer = [0] * BATCH_SIZE
buffer_index = 0
buffer_ready = False

def sample_adc(timer):
    """Timer interrupt callback to sample the ADC. This needs to be fast."""
    global buffer_index, buffer_ready
    if not buffer_ready:
        data_buffer[buffer_index] = adc.read_u16()
        buffer_index += 1
        if buffer_index >= BATCH_SIZE:
            buffer_ready = True
            buffer_index = 0

def main():
    """Main function - simplified without asyncio to avoid conflicts."""
    global buffer_ready
    
    # Wait a moment for USB serial to stabilize
    time.sleep(2)
    
    # Clear any existing output
    print()  # Single newline to clear
    
    print("Pi Pico 5kHz ADC Reader Started")
    print(f"Sampling GPIO{ADC_PIN} at {SAMPLE_RATE_HZ} Hz")
    print(f"Sending batches of {BATCH_SIZE} samples")
    
    # Start the sampling timer
    timer = machine.Timer()
    timer.init(freq=SAMPLE_RATE_HZ, mode=machine.Timer.PERIODIC, callback=sample_adc)
    
    print("READY")
    
    try:
        while True:
            if buffer_ready:
                # Copy buffer quickly to minimize interrupt conflicts
                send_buf = data_buffer[:]  # Fast copy
                buffer_ready = False
                
                # Send the batch header and data
                print(f"BATCH,{BATCH_SIZE}")
                for val in send_buf:
                    print(val)
                print("END_BATCH")
            
            # Small delay to prevent overwhelming the serial port
            time.sleep_ms(5)
            
    except KeyboardInterrupt:
        # Clean shutdown
        timer.deinit()
        print("\nProgram stopped by user")
    except Exception as e:
        # Clean shutdown on any error
        timer.deinit()
        print(f"\nError occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main() 