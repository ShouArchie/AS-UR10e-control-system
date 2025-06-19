import serial
import numpy as np
import time
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys

# === CONFIGURATION ===
SERIAL_PORT = 'COM4'
BAUD_RATE = 500000   # Reliable high-speed baud rate
BUFFER_SIZE = 6000   # Buffer for 12kHz data  
BATCH_SIZE = 10      # Match Arduino batch size (smaller for continuity)
SAMPLE_RATE = 12000  # 12kHz reliable target rate

class PiezoVisualizer:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        
        # Data buffers
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.time_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Statistics and timing
        self.total_samples = 0
        self.batch_count = 0
        self.start_time = time.time()
        self.first_arduino_timestamp = None  # For timing synchronization
        self.debug_counter = 0  # For debugging data reception
        
        # Setup serial connection
        self.setup_serial()
        
        # Setup GUI
        self.setup_gui()
        
        # Setup timer for data reading - extreme speed
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1)  # Update every 1ms for 30kHz extreme speed
        
    def setup_serial(self):
        """Initialize serial connection"""
        print("Connecting to Arduino...")
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(3)
        
        # Wait for Arduino ready signal
        print("Waiting for Arduino...")
        timeout_count = 0
        while True:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if "EXTREME_SPEED_PIEZO_READY" in line:
                    print("Arduino ready for reliable 12kHz continuous sampling!")
                    break
                timeout_count += 1
                if timeout_count > 50:  # 5 seconds timeout
                    print("Timeout waiting for Arduino. Continuing anyway...")
                    break
            except Exception as e:
                print(f"Error waiting for Arduino: {e}")
                timeout_count += 1
                if timeout_count > 50:
                    break
    
    def setup_gui(self):
        """Setup the GUI with multiple plots"""
        self.win = pg.GraphicsLayoutWidget(show=True, title="High-Speed Piezo Visualization")
        self.win.resize(1400, 800)
        
        # Time domain plot - recent data
        self.plot1 = self.win.addPlot(title="Recent Piezo Data (Time Domain)", row=0, col=0)
        self.plot1.setLabel('left', 'Voltage', 'V')
        self.plot1.setLabel('bottom', 'Time', 's')
        self.plot1.addLegend()
        self.curve1 = self.plot1.plot(pen='b', name='Piezo Signal')
        self.plot1.enableAutoRange()  # Auto-scale to show variations better
        
        # Time domain plot - zoomed
        self.plot2 = self.win.addPlot(title="Zoomed View (Last 1000 samples)", row=0, col=1)
        self.plot2.setLabel('left', 'Voltage', 'V')
        self.plot2.setLabel('bottom', 'Sample')
        self.curve2 = self.plot2.plot(pen='r')
        self.plot2.enableAutoRange()  # Auto-scale to show small variations
        
        # Frequency domain plot
        self.win.nextRow()
        self.plot3 = self.win.addPlot(title="Frequency Spectrum", row=1, col=0)
        self.plot3.setLabel('left', 'Magnitude')
        self.plot3.setLabel('bottom', 'Frequency', 'Hz')
        self.curve3 = self.plot3.plot(pen='g')
        self.plot3.setXRange(0, SAMPLE_RATE // 2)
        self.plot3.setLogMode(False, True)  # Log scale on Y axis
        
        # Statistics text
        self.stats_text = self.win.addPlot(title="Statistics", row=1, col=1)
        self.stats_text.hideAxis('left')
        self.stats_text.hideAxis('bottom')
        self.stats_label = pg.TextItem(text="", color='white')
        self.stats_text.addItem(self.stats_label)
        
    def read_arduino_batch(self):
        """Read ASCII batch data with timestamps from Arduino"""
        try:
            line = self.ser.readline().decode('utf-8').strip()
            if line.startswith("BATCH:"):
                # Parse: BATCH:timestamp:value1,value2,value3,...
                parts = line.split(":", 2)  # Split into max 3 parts
                if len(parts) != 3:
                    return None
                
                first_timestamp = int(parts[1])
                data_str = parts[2]
                
                # Convert raw ADC values to voltages
                raw_values = [int(x) for x in data_str.split(',')]
                voltages = [(x * 5.0) / 1023.0 for x in raw_values]
                
                # Return voltages and timing info
                return {
                    'voltages': voltages, 
                    'first_timestamp': first_timestamp, 
                    'sample_interval': 1000000 // SAMPLE_RATE
                }
            
        except Exception as e:
            print(f"Error reading ASCII batch: {e}")
        return None
    
    def update_data(self):
        """Update data and plots"""
        # Read available batches for continuous 15kHz sampling
        batches_read = 0
        while batches_read < 15:  # Process multiple small batches per update
            batch_data = self.read_arduino_batch()
            if batch_data is None:
                break
                
            # Extract voltage data and timing info
            voltages = batch_data['voltages']
            first_timestamp = batch_data['first_timestamp']
            sample_interval = batch_data['sample_interval']
            
            # Synchronize timing on first batch
            if self.first_arduino_timestamp is None:
                self.first_arduino_timestamp = first_timestamp
                self.start_time = time.time()
            
            # Add to buffers with accurate timestamps
            for i, voltage in enumerate(voltages):
                self.data_buffer.append(voltage)
                # Calculate accurate relative time using Arduino timestamp
                sample_timestamp_micros = first_timestamp + (i * sample_interval)
                relative_micros = sample_timestamp_micros - self.first_arduino_timestamp
                sample_time_seconds = relative_micros / 1000000.0
                self.time_buffer.append(sample_time_seconds)
            
            self.total_samples += len(voltages)
            self.batch_count += 1
            batches_read += 1
            
            # Optional: Debug output every 500 batches (commented out for clean interface)
            # self.debug_counter += 1
            # if self.debug_counter % 500 == 0:
            #     print(f"Received batch {self.batch_count}: {len(voltages)} samples, rate: {self.total_samples/(time.time()-self.start_time):.0f} Hz")
        
        # Update plots if we have data
        if len(self.data_buffer) > 0:
            self.update_plots()
    
    def update_plots(self):
        """Update all plots with current data"""
        current_data = np.array(list(self.data_buffer))
        current_time = np.array(list(self.time_buffer))
        
        # Time domain plot - recent data (last 2000 samples)
        if len(current_data) > 2000:
            recent_data = current_data[-2000:]
            recent_time = current_time[-2000:]
        else:
            recent_data = current_data
            recent_time = current_time
        
        self.curve1.setData(recent_time, recent_data)
        
        # Zoomed time domain plot (last 1000 samples)
        if len(current_data) > 1000:
            zoomed_data = current_data[-1000:]
            zoomed_indices = np.arange(len(zoomed_data))
        else:
            zoomed_data = current_data
            zoomed_indices = np.arange(len(zoomed_data))
        
        self.curve2.setData(zoomed_indices, zoomed_data)
        
        # Frequency domain plot
        if len(current_data) >= 1024:
            # Use last 1024 samples for frequency analysis (good for 10kHz)
            fft_data = np.fft.fft(current_data[-1024:])
            freqs = np.fft.fftfreq(1024, 1/SAMPLE_RATE)
            magnitude = np.abs(fft_data[:512])  # Positive frequencies only
            
            # Remove DC component and apply some smoothing
            magnitude[0] = 0
            self.curve3.setData(freqs[:512], magnitude)
        
        # Update statistics
        self.update_statistics(current_data)
    
    def update_statistics(self, data):
        """Update statistics display"""
        elapsed_time = time.time() - self.start_time
        actual_rate = self.total_samples / elapsed_time if elapsed_time > 0 else 0
        
        if len(data) > 0:
            stats_text = f"""Total Samples: {self.total_samples:,}
Batches: {self.batch_count:,}
Elapsed: {elapsed_time:.1f}s
Rate: {actual_rate:.0f} Hz
Buffer: {len(data):,} samples

Current: {data[-1]:.4f} V
Average: {np.mean(data):.4f} V
RMS: {np.sqrt(np.mean(data**2)):.4f} V
Std Dev: {np.std(data):.4f} V
Min: {np.min(data):.4f} V
Max: {np.max(data):.4f} V
P-P: {np.max(data) - np.min(data):.4f} V"""
        else:
            stats_text = "Waiting for data..."
        
        self.stats_label.setText(stats_text)
    
    def run(self):
        """Start the application"""
        print("Starting high-speed visualization...")
        print("Close window to exit.")
        
        try:
            self.app.exec_()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.timer.stop()
        if hasattr(self, 'ser'):
            self.ser.close()
        
        # Final statistics
        elapsed_time = time.time() - self.start_time
        actual_rate = self.total_samples / elapsed_time if elapsed_time > 0 else 0
        print(f"\n=== Final Statistics ===")
        print(f"Total samples: {self.total_samples:,}")
        print(f"Average rate: {actual_rate:.0f} Hz")
        print(f"Total time: {elapsed_time:.1f}s")

if __name__ == "__main__":
    visualizer = PiezoVisualizer()
    visualizer.run() 