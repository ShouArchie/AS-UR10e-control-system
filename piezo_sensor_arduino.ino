const int piezoPin = A0;    // Piezo sensor connected to analog pin A0
const int sampleDelay = 50; // Read every 50ms (20Hz)

void setup() {
  Serial.begin(9600);  // Initialize serial communication at 9600 baud
  pinMode(piezoPin, INPUT);
}

void loop() {
  // Read the piezo sensor value (0-1023)
  int sensorValue = analogRead(piezoPin);
  
  // Send the raw value over serial
  Serial.println(sensorValue);
  
  // Wait before next reading
  delay(sampleDelay);
}
