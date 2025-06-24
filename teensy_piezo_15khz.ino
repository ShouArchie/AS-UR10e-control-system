/*
 * Teensy 4.0 Dual Piezo Sensor Data Acquisition - High Performance Edition
 * 
 * Reads analog input from pins A0 and A1 at 18 kHz sampling rate (9kHz per channel)
 * Sends raw 12-bit ADC values (0-4095) over USB Serial in interleaved format
 * Uses IntervalTimer for precise timing and buffering for efficiency
 * 
 * Hardware:
 * - Teensy 4.0
 * - Piezo sensor 1 with 20x opamp circuit connected to A0
 * - Piezo sensor 2 with 20x opamp circuit connected to A1
 * - Input voltage range: 0-3.3V per channel
 * 
 * Serial Protocol:
 * - Baud rate: 921600 (optimized for 18kHz dual-channel data rate - 72KB/s)
 * - Data format: Binary, 4 bytes per sample pair (A0, A1) little-endian
 * - Interleaved format: A0_sample, A1_sample, A0_sample, A1_sample...
 * - Buffered transmission for efficiency
 */

#define ANALOG_PIN_A0 A0
#define ANALOG_PIN_A1 A1
#define SAMPLE_RATE_HZ 18000  // Total sample rate (9kHz per channel) - Optimized for USB stability
#define SAMPLE_INTERVAL_US (1000000 / SAMPLE_RATE_HZ)  // 50 microseconds

// Buffer settings for efficient transmission (dual channel)
#define BUFFER_SIZE 100  // Buffer 100 sample pairs (200 values total) before sending
#define ADC_RESOLUTION 12  // 12-bit ADC (0-4095)

// Data buffer for interleaved A0,A1,A0,A1... format
uint16_t sampleBuffer[BUFFER_SIZE * 2];  // Double size for dual channel
volatile int bufferIndex = 0;
volatile bool bufferReady = false;

// Performance monitoring
volatile unsigned long sampleCount = 0;
unsigned long lastStatsTime = 0;

// Use IntervalTimer for precise timing
IntervalTimer sampleTimer;

void setup() {
  // Configure analog inputs
  pinMode(ANALOG_PIN_A0, INPUT);
  pinMode(ANALOG_PIN_A1, INPUT);
  
  // Set ADC resolution to 12 bits
  analogReadResolution(ADC_RESOLUTION);
  
  // Teensy 4.0 uses 3.3V reference by default (no need to set analogReference)
  // Available options: DEFAULT (3.3V), INTERNAL (1.2V)
  // We want the default 3.3V for full voltage range
  
  // Optimize ADC settings for speed
  analogReadAveraging(1);  // No averaging for maximum speed
  
  // Initialize serial communication  
  Serial.begin(921600);  // Higher baud rate for 20kHz dual-channel sampling
  
  // Wait for serial connection (optional - comment out for standalone operation)
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Send startup message
  Serial.println("Teensy 4.0 Dual Piezo Sensor - High Performance 18kHz ADC");
  Serial.println("Dual Channel A0/A1 - Optimized with IntervalTimer and buffering");
  Serial.println("Data format: Interleaved A0,A1,A0,A1...");
  Serial.println("Starting dual-channel data acquisition...");
  delay(1000);
  
  // Clear the startup text from serial buffer
  Serial.flush();
  
  // Start precise timing with IntervalTimer
  sampleTimer.begin(sampleISR, SAMPLE_INTERVAL_US);
  
  lastStatsTime = millis();
}

void sampleISR() {
  // This runs every ~55.6 microseconds (18 kHz total, 9kHz per channel)
  // Read analog values from both A0 and A1
  uint16_t adcValueA0 = analogRead(ANALOG_PIN_A0);
  uint16_t adcValueA1 = analogRead(ANALOG_PIN_A1);
  
  // Store in buffer in interleaved format: A0, A1, A0, A1...
  sampleBuffer[bufferIndex] = adcValueA0;
  sampleBuffer[bufferIndex + 1] = adcValueA1;
  bufferIndex += 2;  // Increment by 2 for dual channel
  sampleCount++;  // Count sample pairs
  
  // Check if buffer is full
  if (bufferIndex >= (BUFFER_SIZE * 2)) {
    bufferIndex = 0;
    bufferReady = true;
  }
}

void loop() {
  // Send buffered data when ready
  if (bufferReady) {
    // Temporarily disable interrupts for consistent data
    noInterrupts();
    bufferReady = false;
    
    // Send entire buffer at once for efficiency
    // Buffer contains BUFFER_SIZE * 2 uint16_t values = BUFFER_SIZE * 4 bytes
    Serial.write((uint8_t*)sampleBuffer, BUFFER_SIZE * 4);
    
    interrupts();
  }
  
  // Optional: Print performance statistics every 5 seconds
  // DISABLED to prevent text/binary data mixing which causes spikes
  // unsigned long currentTime = millis();
  // if (currentTime - lastStatsTime >= 5000) {
  //   printPerformanceStats();
  //   lastStatsTime = currentTime;
  // }
  
  // Small delay to prevent overwhelming the main loop
  delayMicroseconds(10);
}

void printPerformanceStats() {
  // Temporarily disable sampling for stats
  noInterrupts();
  unsigned long samples = sampleCount;
  sampleCount = 0;  // Reset counter
  interrupts();
  
  // Calculate actual sample rate (sample pairs per second)
  float actualRate = samples / 5.0;  // Sample pairs per second over 5 second period
  float totalRate = actualRate * 2;  // Total samples per second (both channels)
  
  Serial.print("Performance: ");
  Serial.print(totalRate, 1);
  Serial.print(" Hz total (");
  Serial.print(actualRate, 1);
  Serial.print(" Hz per channel) | Target: ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(" Hz | Efficiency: ");
  Serial.print((totalRate / SAMPLE_RATE_HZ) * 100.0, 1);
  Serial.println("%");
}

/*
 * Data Format Explanation:
 * 
 * The buffer contains interleaved 16-bit ADC values:
 * [A0_0, A1_0, A0_1, A1_1, A0_2, A1_2, ...]
 * 
 * When transmitted as bytes (little-endian):
 * [A0_0_LSB, A0_0_MSB, A1_0_LSB, A1_0_MSB, A0_1_LSB, A0_1_MSB, A1_1_LSB, A1_1_MSB, ...]
 * 
 * Python receiver should parse as:
 * - Read 4 bytes at a time
 * - Convert to 2 uint16_t values (A0, A1)
 * - Process A0 and A1 samples accordingly
 * 
 * Alternative ultra-high performance version using DMA (advanced):
 * 
 * #include <DMAChannel.h>
 * 
 * DMAChannel dma;
 * volatile uint16_t adcBuffer[BUFFER_SIZE * 2];
 * 
 * void setupDMA() {
 *   // Configure DMA for continuous dual-channel ADC transfers
 *   dma.source(ADC0_RA);
 *   dma.destinationBuffer(adcBuffer, BUFFER_SIZE * 4);
 *   dma.transferSize(2);
 *   dma.transferCount(BUFFER_SIZE * 2);
 *   dma.interruptAtCompletion();
 *   dma.attachInterrupt(dmaComplete);
 *   dma.enable();
 * }
 * 
 * void dmaComplete() {
 *   Serial.write((uint8_t*)adcBuffer, BUFFER_SIZE * 4);
 *   dma.clearInterrupt();
 * }
 */ 