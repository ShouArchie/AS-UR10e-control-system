/*
 * Teensy 4.0 Piezo Sensor Data Acquisition - High Performance Edition
 * 
 * Reads analog input from pin A0 at 20 kHz sampling rate
 * Sends raw 12-bit ADC values (0-4095) over USB Serial
 * Uses IntervalTimer for precise timing and buffering for efficiency
 * 
 * Hardware:
 * - Teensy 4.0
 * - Piezo sensor #1 with 20x opamp circuit connected to A0
 * - Piezo sensor #2 with 20x opamp circuit connected to A1
 * - Input voltage range: 0-3.3V per channel
 * 
 * Serial Protocol:
 * - Baud rate: 921600 (optimized for 20kHz data rate)
 * - Data format: Binary, 4 bytes per sample pair (A0, A1 interleaved, little-endian)
 * - Buffered transmission for efficiency
 */

#define ANALOG_PIN_1 A0
#define ANALOG_PIN_2 A1
#define SAMPLE_RATE_HZ 20000
#define SAMPLE_INTERVAL_US (1000000 / SAMPLE_RATE_HZ)  // 50 microseconds

// Buffer settings for efficient transmission
#define BUFFER_SIZE 100  // Buffer 100 samples before sending (per channel)
#define ADC_RESOLUTION 12  // 12-bit ADC (0-4095)

// Data buffers for dual channel
uint16_t sampleBuffer_A0[BUFFER_SIZE];
uint16_t sampleBuffer_A1[BUFFER_SIZE];
volatile int bufferIndex = 0;
volatile bool bufferReady = false;

// Performance monitoring
volatile unsigned long sampleCount = 0;
unsigned long lastStatsTime = 0;

// Use IntervalTimer for precise timing
IntervalTimer sampleTimer;

void setup() {
  // Configure analog inputs
  pinMode(ANALOG_PIN_1, INPUT);
  pinMode(ANALOG_PIN_2, INPUT);
  
  // Set ADC resolution to 12 bits
  analogReadResolution(ADC_RESOLUTION);
  
  // Teensy 4.0 uses 3.3V reference by default (no need to set analogReference)
  // Available options: DEFAULT (3.3V), INTERNAL (1.2V)
  // We want the default 3.3V for full voltage range
  
  // Optimize ADC settings for speed
  analogReadAveraging(1);  // No averaging for maximum speed
  
  // Initialize serial communication  
  Serial.begin(921600);  // Higher baud rate for 20kHz sampling
  
  // Wait for serial connection (optional - comment out for standalone operation)
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Send startup message
  Serial.println("Teensy 4.0 Dual Piezo Sensor - High Performance 20kHz ADC");
  Serial.println("Reading A0 and A1 simultaneously with IntervalTimer and buffering");
  Serial.println("Starting dual channel data acquisition...");
  delay(1000);
  
  // Clear the startup text from serial buffer
  Serial.flush();
  
  // Start precise timing with IntervalTimer
  sampleTimer.begin(sampleISR, SAMPLE_INTERVAL_US);
  
  lastStatsTime = millis();
}

void sampleISR() {
  // This runs every 50 microseconds (20 kHz)
  // Read analog values from both A0 and A1
  uint16_t adcValue_A0 = analogRead(ANALOG_PIN_1);
  uint16_t adcValue_A1 = analogRead(ANALOG_PIN_2);
  
  // Store in buffers
  sampleBuffer_A0[bufferIndex] = adcValue_A0;
  sampleBuffer_A1[bufferIndex] = adcValue_A1;
  bufferIndex++;
  sampleCount++;
  
  // Check if buffer is full
  if (bufferIndex >= BUFFER_SIZE) {
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
    
    // Send both buffers interleaved (A0, A1, A0, A1, ...)
    for (int i = 0; i < BUFFER_SIZE; i++) {
      Serial.write((uint8_t*)&sampleBuffer_A0[i], 2);  // Send A0 sample (2 bytes)
      Serial.write((uint8_t*)&sampleBuffer_A1[i], 2);  // Send A1 sample (2 bytes)
    }
    
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
  
  // Calculate actual sample rate
  float actualRate = samples / 5.0;  // Samples per second over 5 second period
  
  Serial.print("Performance: ");
  Serial.print(actualRate, 1);
  Serial.print(" Hz (target: ");
  Serial.print(SAMPLE_RATE_HZ);
  Serial.print(" Hz) | Efficiency: ");
  Serial.print((actualRate / SAMPLE_RATE_HZ) * 100.0, 1);
  Serial.println("%");
}

/*
 * Alternative ultra-high performance version using DMA (advanced):
 * 
 * #include <DMAChannel.h>
 * 
 * DMAChannel dma;
 * volatile uint16_t adcBuffer[BUFFER_SIZE];
 * 
 * void setupDMA() {
 *   // Configure DMA for continuous ADC transfers
 *   dma.source(ADC0_RA);
 *   dma.destinationBuffer(adcBuffer, BUFFER_SIZE * 2);
 *   dma.transferSize(2);
 *   dma.transferCount(BUFFER_SIZE);
 *   dma.interruptAtCompletion();
 *   dma.attachInterrupt(dmaComplete);
 *   dma.enable();
 * }
 * 
 * void dmaComplete() {
 *   Serial.write((uint8_t*)adcBuffer, BUFFER_SIZE * 2);
 *   dma.clearInterrupt();
 * }
 */ 