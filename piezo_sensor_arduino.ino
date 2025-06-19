// ADC Prescaler definitions (must be defined before use)
#define PS_8   (1 << ADPS1) | (1 << ADPS0)  // Fastest prescaler for maximum speed
#define PS_16  (1 << ADPS2)
#define PS_32  (1 << ADPS2) | (1 << ADPS0)
#define PS_64  (1 << ADPS2) | (1 << ADPS1)
#define PS_128 (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0)

const int piezoPin = A0;    // Piezo sensor connected to analog pin A0
const float referenceVoltage = 5.0; // Arduino reference voltage (5V for Uno/Nano, 3.3V for some boards)

// Optimized continuous sampling configuration
const int BATCH_SIZE = 10;   // Tiny batches for near-continuous sampling
const int SAMPLE_RATE = 12000; // Conservative 12kHz target for reliability
const unsigned long SAMPLE_INTERVAL_MICROS = 1000000 / SAMPLE_RATE; // 83 microseconds

// Buffer for storing samples with timestamps
int sampleBuffer[BATCH_SIZE];
unsigned long timestampBuffer[BATCH_SIZE];
int bufferIndex = 0;
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(500000);  // Reliable high-speed baud rate
  pinMode(piezoPin, INPUT);
  
  // Configure ADC for absolute maximum speed
  // Set prescaler to 8 and enable high-speed mode
  ADCSRA &= ~PS_128;  // Clear all prescaler bits
  ADCSRA |= PS_8;     // Set prescaler to 8 for maximum speed
  ADCSRA |= (1 << ADEN);  // Enable ADC
  
  // Send ready message - no delay needed for ASCII protocol
  Serial.println("EXTREME_SPEED_PIEZO_READY");
}

void loop() {
  unsigned long currentTime = micros();
  
  // Precise timing check for 30kHz sampling
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MICROS) {
    // Ultra-fast ADC read using direct port manipulation
    ADMUX = (1 << REFS0);  // AVcc reference, ADC0 (A0)
    ADCSRA |= (1 << ADSC); // Start conversion
    while (ADCSRA & (1 << ADSC)); // Wait for conversion (takes ~13 cycles at prescaler 8)
    
    int sensorValue = ADC; // Read ADC result register directly
    
    // Store sample and timestamp in buffer
    sampleBuffer[bufferIndex] = sensorValue;
    timestampBuffer[bufferIndex] = currentTime;
    bufferIndex++;
    
    // Update timing immediately to maintain precise intervals
    lastSampleTime = currentTime;
    
    // When buffer is full, send batch
    if (bufferIndex >= BATCH_SIZE) {
      sendBatch();
      bufferIndex = 0;
    }
  }
}

void sendBatch() {
  // ASCII format with timestamp for accurate timing
  Serial.print("BATCH:");
  Serial.print(timestampBuffer[0]);  // First sample timestamp
  Serial.print(":");
  
  // Send raw ADC values as comma-separated integers
  for (int i = 0; i < BATCH_SIZE; i++) {
    Serial.print(sampleBuffer[i]);
    if (i < BATCH_SIZE - 1) {
      Serial.print(",");
    }
  }
  Serial.println();
}
 
 