import serial
import numpy as np
import time
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from collections import deque

# === CONFIGURATION ===
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600
BUFFER_SIZE = 100  # Number of samples to keep in the buffer
TRAINING_SIZE = 500  # Number of samples to collect for initial training
ANOMALY_THRESHOLD = -0.5  # Isolation Forest threshold (lower = more anomalies)

def read_arduino_data(ser):
    """Read a single data point from Arduino"""
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            return float(line)
    except (ValueError, UnicodeDecodeError):
        pass
    return None

def main():
    print("Starting Piezo Electric Sensor Anomaly Detection...")
    
    # Initialize serial connection to Arduino
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow time for Arduino connection to stabilize
    
    # Initialize data structures
    data_buffer = deque(maxlen=BUFFER_SIZE)
    all_data = []
    anomaly_scores = []
    
    # Collect initial training data
    print(f"Collecting {TRAINING_SIZE} samples for training...")
    while len(all_data) < TRAINING_SIZE:
        value = read_arduino_data(ser)
        if value is not None:
            all_data.append(value)
            print(f"Training data: {len(all_data)}/{TRAINING_SIZE}", end='\r')
    
    # Train initial model
    print("\nTraining isolation forest model...")
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(np.array(all_data).reshape(-1, 1))
    print("Model trained!")
    
    # Setup plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    line1, = ax1.plot([], [], 'b-')
    line2, = ax2.plot([], [], 'r-')
    ax1.set_title('Piezo Sensor Readings')
    ax1.set_ylabel('Voltage')
    ax2.set_title('Anomaly Scores')
    ax2.set_ylabel('Score')
    ax2.axhline(y=ANOMALY_THRESHOLD, color='g', linestyle='--')
    
    print("Monitoring for disturbances... Press Ctrl+C to exit.")
    try:
        while True:
            value = read_arduino_data(ser)
            if value is not None:
                # Add to data structures
                data_buffer.append(value)
                all_data.append(value)
                
                # Calculate anomaly score
                score = model.decision_function(np.array([value]).reshape(1, -1))[0]
                anomaly_scores.append(score)
                
                # Check for anomaly
                if score < ANOMALY_THRESHOLD:
                    print(f"ANOMALY DETECTED! Value: {value:.2f}, Score: {score:.4f}")
                
                # Periodically retrain the model with recent data
                if len(all_data) % 100 == 0:
                    recent_data = all_data[-1000:] if len(all_data) > 1000 else all_data
                    model.fit(np.array(recent_data).reshape(-1, 1))
                    print("Model retrained with recent data")
                
                # Update plots
                line1.set_data(range(len(data_buffer)), list(data_buffer))
                line2.set_data(range(len(anomaly_scores[-BUFFER_SIZE:])), anomaly_scores[-BUFFER_SIZE:])
                
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                plt.pause(0.01)
                
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    finally:
        ser.close()
        plt.ioff()
        plt.show()
        print("Serial connection closed.")

if __name__ == "__main__":
    main() 