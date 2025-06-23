/*
 * Teensy 4.0 Piezo Sensor Data Acquisition
 * 
 * Reads analog input from pin A0 at 15 kHz sampling rate
 * Sends raw 12-bit ADC values (0-4095) over USB Serial
 * 
 * Hardware:
 * - Teensy 4.0
 * - Piezo sensor with 20x opamp circuit connected to A0
 * - Input voltage range: 0-3.3V
 * 
 * Serial Protocol:
 * - Baud rate: 500000 (to handle 15k samples/sec)
 * - Data format: Binary, 2 bytes per sample (little-endian)
 * - Each sample is a 16-bit value containing 12-bit ADC reading
 */

#define ANALOG_PIN A0
#define SAMPLE_RATE_HZ 15000
#define SAMPLE_INTERVAL_US (1000000 / SAMPLE_RATE_HZ)  // 66.67 microseconds

// Timing variables
unsigned long lastSampleTime = 0;
unsigned long sampleInterval = SAMPLE_INTERVAL_US;

// ADC resolution
const int ADC_RESOLUTION = 12;  // 12-bit ADC (0-4095)

void setup() {
  // Configure analog input
  pinMode(ANALOG_PIN, INPUT);
  
  // Set ADC resolution to 12 bits
  analogReadResolution(ADC_RESOLUTION);
  
  // Initialize serial communication
  Serial.begin(500000);  // High baud rate for 15kHz data stream
  
  // Wait for serial connection (optional - comment out for standalone operation)
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Send startup message
  Serial.println("Teensy 4.0 Piezo Sensor - 15kHz ADC");
  Serial.println("Starting data acquisition...");
  delay(1000);
  
  // Initialize timing
  lastSampleTime = micros();
}

void loop() {
  unsigned long currentTime = micros();
  
  // Check if it's time to sample
  if (currentTime - lastSampleTime >= sampleInterval) {
    // Read analog value from A0
    uint16_t adcValue = analogRead(ANALOG_PIN);
    
    // Send raw ADC value as binary data (2 bytes, little-endian)
    Serial.write((byte)(adcValue & 0xFF));        // Low byte
    Serial.write((byte)((adcValue >> 8) & 0xFF)); // High byte
    
    // Update timing for next sample
    lastSampleTime = currentTime;
  }
  
  // Optional: Add small delay to prevent overwhelming the system
  // delayMicroseconds(1);
}

/*
 * Alternative implementation using IntervalTimer for more precise timing:
 * 
 * IntervalTimer sampleTimer;
 * 
 * void sampleISR() {
 *   uint16_t adcValue = analogRead(ANALOG_PIN);
 *   Serial.write((byte)(adcValue & 0xFF));
 *   Serial.write((byte)((adcValue >> 8) & 0xFF));
 * }
 * 
 * In setup(), replace timing code with:
 * sampleTimer.begin(sampleISR, SAMPLE_INTERVAL_US);
 */ 