import numpy as np
import serial
import serial.tools.list_ports
import threading
import time
from collections import deque
from scipy import signal
import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QSpinBox, QCheckBox, QGroupBox, QGridLayout
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont

class HighFreqDataCollector:
    def __init__(self, max_samples=1000000):  # 10 seconds at 100kHz
        # Circular buffers using NumPy for speed
        self.max_samples = max_samples
        self.voltage_buffer = np.zeros(max_samples, dtype=np.float32)
        self.time_buffer = np.zeros(max_samples, dtype=np.float64)
        self.write_index = 0
        self.total_samples = 0
        self.buffer_full = False
        
        # Statistics
        self.sample_rate = 100000  # Hz
        self.last_update = time.time()
        self.batch_count = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Serial connection
        self.serial_port = None
        self.running = False
        self.thread = None
        
    def find_pico_port(self):
        """Find Raspberry Pi Pico port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if hasattr(port, 'vid') and port.vid == 0x2E8A:  # Pico VID
                return port.device
        # Fallback to first available port
        if ports:
            return ports[0].device
        return None
    
    def connect(self, port=None):
        """Connect to serial port"""
        if port is None:
            port = self.find_pico_port()
        
        if port is None:
            raise Exception("No serial port found")
        
        try:
            # Close any existing connection first
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                time.sleep(0.5)  # Wait for port to be released
            
            # Try to open with more robust settings
            self.serial_port = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=0.1,
                write_timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            
            # Clear any existing data
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            print(f"Connected to {port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def add_batch(self, voltages, start_time):
        """Add batch of voltage data efficiently"""
        if len(voltages) == 0:
            return
            
        with self.lock:
            n_samples = len(voltages)
            
            # Create time array for this batch
            dt = 1.0 / self.sample_rate
            times = start_time + np.arange(n_samples) * dt
            
            # Handle circular buffer wraparound
            end_index = self.write_index + n_samples
            
            if end_index <= self.max_samples:
                # Simple case: no wraparound
                self.voltage_buffer[self.write_index:end_index] = voltages
                self.time_buffer[self.write_index:end_index] = times
            else:
                # Wraparound case
                first_part = self.max_samples - self.write_index
                self.voltage_buffer[self.write_index:] = voltages[:first_part]
                self.time_buffer[self.write_index:] = times[:first_part]
                
                remaining = n_samples - first_part
                self.voltage_buffer[:remaining] = voltages[first_part:]
                self.time_buffer[:remaining] = times[first_part:]
                
                self.buffer_full = True
            
            self.write_index = (self.write_index + n_samples) % self.max_samples
            self.total_samples += n_samples
            self.batch_count += 1
            self.last_update = time.time()
    
    def get_recent_data(self, n_samples=10000):
        """Get most recent n_samples efficiently"""
        with self.lock:
            if not self.buffer_full and self.write_index < n_samples:
                # Not enough data yet
                return self.time_buffer[:self.write_index], self.voltage_buffer[:self.write_index]
            
            if not self.buffer_full:
                # Buffer not full, get last n_samples
                start_idx = max(0, self.write_index - n_samples)
                return self.time_buffer[start_idx:self.write_index], self.voltage_buffer[start_idx:self.write_index]
            
            # Buffer is full, get data in circular order
            if self.write_index >= n_samples:
                start_idx = self.write_index - n_samples
                return self.time_buffer[start_idx:self.write_index], self.voltage_buffer[start_idx:self.write_index]
            else:
                # Wraparound case
                first_part = n_samples - self.write_index
                times = np.concatenate([
                    self.time_buffer[-first_part:],
                    self.time_buffer[:self.write_index]
                ])
                voltages = np.concatenate([
                    self.voltage_buffer[-first_part:],
                    self.voltage_buffer[:self.write_index]
                ])
                return times, voltages
    
    def get_stats(self):
        """Get current statistics"""
        with self.lock:
            if self.total_samples == 0:
                return {"samples": 0, "rate": 0, "avg": 0, "min": 0, "max": 0, "batches": 0}
            
            # Get recent data for stats
            recent_times, recent_voltages = self.get_recent_data(min(10000, self.total_samples))
            
            if len(recent_voltages) == 0:
                return {"samples": 0, "rate": 0, "avg": 0, "min": 0, "max": 0, "batches": 0}
            
            # Calculate actual sample rate
            if len(recent_times) > 1:
                actual_rate = len(recent_times) / (recent_times[-1] - recent_times[0])
            else:
                actual_rate = 0
            
            return {
                "samples": self.total_samples,
                "rate": actual_rate,
                "avg": np.mean(recent_voltages),
                "min": np.min(recent_voltages),
                "max": np.max(recent_voltages),
                "batches": self.batch_count
            }
    
    def data_collection_thread(self):
        """High-speed data collection thread"""
        buffer = b""
        current_batch = []
        in_batch = False
        batch_start_time = time.time()
        consecutive_errors = 0
        
        print("Data collection thread started")
        
        while self.running:
            try:
                # Check if serial port is still valid
                if not self.serial_port or not self.serial_port.is_open:
                    print("Serial port closed, stopping data collection")
                    break
                
                # Read available data
                waiting = self.serial_port.in_waiting
                if waiting > 0:
                    data = self.serial_port.read(waiting)
                    consecutive_errors = 0  # Reset error counter
                else:
                    data = self.serial_port.read(1)  # Try to read at least 1 byte
                
                if not data:
                    time.sleep(0.001)  # 1ms sleep if no data
                    continue
                
                buffer += data
                
                # Process complete lines
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    line = line.decode('utf-8', errors='ignore').strip()
                    
                    # Debug: print first few lines to see what we're getting
                    if self.total_samples < 10:
                        print(f"Received: {line}")
                    
                    if line.startswith("BATCH_START:"):
                        current_batch = []
                        in_batch = True
                        batch_start_time = time.time()
                        print(f"Batch started at {batch_start_time}")
                        
                    elif line.startswith("BATCH_END:"):
                        in_batch = False
                        if current_batch:
                            # Convert to numpy array for speed
                            voltages = np.array(current_batch, dtype=np.float32)
                            self.add_batch(voltages, batch_start_time)
                            print(f"Batch ended: {len(current_batch)} samples")
                        else:
                            print("Empty batch received")
                            
                    elif in_batch:
                        try:
                            voltage = float(line)
                            current_batch.append(voltage)
                        except ValueError:
                            if line:  # Only print non-empty invalid lines
                                print(f"Invalid voltage line: '{line}'")
                    
                    elif line.startswith("===") or line.startswith("Status:"):
                        print(f"[PICO] {line}")
                    elif line and not line.startswith("BATCH"):
                        print(f"Unknown line: '{line}'")
                            
            except Exception as e:
                consecutive_errors += 1
                print(f"Data collection error ({consecutive_errors}): {e}")
                
                if consecutive_errors > 10:
                    print("Too many consecutive errors, stopping data collection")
                    break
                    
                time.sleep(0.1)
        
        print("Data collection thread ended")
    
    def start_collection(self):
        """Start data collection thread"""
        if not self.serial_port:
            raise Exception("Not connected to serial port")
        
        self.running = True
        self.thread = threading.Thread(target=self.data_collection_thread, daemon=True)
        self.thread.start()
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial_port:
            self.serial_port.close()

class DataUpdateSignal(QObject):
    update_signal = pyqtSignal()

class HighFreqMonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.collector = HighFreqDataCollector()
        
        # Display parameters
        self.display_samples = 5000  # Show last 5000 points
        self.downsample_factor = 1   # Show every Nth point
        self.update_interval = 50    # Update every 50ms
        
        # Setup PyQtGraph for high performance
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        # Create GUI elements
        self.setup_gui()
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        
        # Data update signal
        self.data_signal = DataUpdateSignal()
        self.data_signal.update_signal.connect(self.update_plot)
        
    def setup_gui(self):
        """Setup GUI elements"""
        self.setWindowTitle("High-Frequency Piezo Monitor (100kHz+)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        
        # Connection controls
        control_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.addItems([p.device for p in serial.tools.list_ports.comports()])
        control_layout.addWidget(self.port_combo)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect)
        control_layout.addWidget(self.connect_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        # Display controls
        control_layout.addWidget(QLabel("  |  Display:"))
        self.display_spin = QSpinBox()
        self.display_spin.setRange(1000, 50000)
        self.display_spin.setValue(5000)
        self.display_spin.setSuffix(" samples")
        control_layout.addWidget(self.display_spin)
        
        control_layout.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 100)
        self.downsample_spin.setValue(1)
        control_layout.addWidget(self.downsample_spin)
        
        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(True)
        control_layout.addWidget(self.auto_scale_cb)
        
        control_layout.addStretch()
        main_layout.addWidget(control_group)
        
        # Statistics panel
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.stats_labels = {}
        stats_items = [
            ("samples", "Samples:"), ("rate", "Rate (Hz):"), ("batches", "Batches:"),
            ("avg", "Avg (V):"), ("min", "Min (V):"), ("max", "Max (V):")
        ]
        
        for i, (key, label) in enumerate(stats_items):
            stats_layout.addWidget(QLabel(label), 0, i*2)
            self.stats_labels[key] = QLabel("0")
            self.stats_labels[key].setStyleSheet("color: blue; font-weight: bold;")
            stats_layout.addWidget(self.stats_labels[key], 0, i*2+1)
        
        main_layout.addWidget(stats_group)
        
        # Plot widget container
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Time domain plot
        self.time_plot = pg.PlotWidget(title="Time Domain - Voltage vs Time")
        self.time_plot.setLabel('left', 'Voltage', units='V')
        self.time_plot.setLabel('bottom', 'Time', units='s')
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.time_plot.setMinimumHeight(300)
        
        # Create time domain curve
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(color='b', width=1))
        
        plot_layout.addWidget(self.time_plot)
        
        # Frequency domain plot
        self.freq_plot = pg.PlotWidget(title="Frequency Domain - FFT")
        self.freq_plot.setLabel('left', 'Magnitude', units='V')
        self.freq_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.freq_plot.setLogMode(y=True)
        self.freq_plot.setMinimumHeight(300)
        
        # Create frequency domain curve
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen(color='r', width=1))
        
        plot_layout.addWidget(self.freq_plot)
        
        main_layout.addWidget(plot_widget)
        
    def connect(self):
        """Connect to serial port"""
        port = self.port_combo.currentText() or None
        if self.collector.connect(port):
            self.connect_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
        
    def start_monitoring(self):
        """Start monitoring"""
        self.collector.start_collection()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start timer for updates
        self.timer.start(self.update_interval)
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.timer.stop()
        self.collector.stop_collection()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def update_display_params(self):
        """Update display parameters"""
        self.display_samples = self.display_spin.value()
        self.downsample_factor = self.downsample_spin.value()
    
    def update_plot(self):
        """Update plot with new data"""
        # Update display parameters
        self.update_display_params()
        
        # Get recent data
        times, voltages = self.collector.get_recent_data(self.display_samples)
        
        if len(times) == 0:
            return
        
        # Downsample for display
        if self.downsample_factor > 1:
            times = times[::self.downsample_factor]
            voltages = voltages[::self.downsample_factor]
        
        # Update time domain plot
        if len(times) > 0:
            self.time_curve.setData(times, voltages)
            
            # Auto-scale if enabled
            if self.auto_scale_cb.isChecked():
                self.time_plot.enableAutoRange()
            else:
                self.time_plot.disableAutoRange()
        
        # Update frequency domain plot (FFT)
        if len(voltages) > 100:  # Need enough samples for meaningful FFT
            try:
                # Use recent data for FFT
                fft_size = min(8192, len(voltages))  # Use up to 8192 points
                fft_data = voltages[-fft_size:]
                
                # Remove DC component
                fft_data = fft_data - np.mean(fft_data)
                
                # Apply window to reduce spectral leakage
                window = signal.windows.hann(len(fft_data))
                fft_data = fft_data * window
                
                # Compute FFT
                freqs = np.fft.fftfreq(len(fft_data), 1/self.collector.sample_rate)
                fft_mag = np.abs(np.fft.fft(fft_data))
                
                # Take positive frequencies only
                pos_freqs = freqs[:len(freqs)//2]
                pos_mag = fft_mag[:len(fft_mag)//2]
                
                # Filter out zero frequencies and very small magnitudes
                valid_mask = (pos_freqs > 0) & (pos_mag > 1e-6)
                pos_freqs = pos_freqs[valid_mask]
                pos_mag = pos_mag[valid_mask]
                
                # Update frequency plot
                if len(pos_freqs) > 0:
                    self.freq_curve.setData(pos_freqs, pos_mag)
                    
            except Exception as e:
                print(f"FFT error: {e}")
        
        # Update statistics
        stats = self.collector.get_stats()
        for key, value in stats.items():
            if key in self.stats_labels:
                if key == "rate":
                    self.stats_labels[key].setText(f"{value:.0f}")
                elif key in ["avg", "min", "max"]:
                    self.stats_labels[key].setText(f"{value:.4f}")
                else:
                    self.stats_labels[key].setText(f"{value}")
    
    def run(self):
        """Run the GUI"""
        self.show()

if __name__ == "__main__":
    print("High-Frequency Piezo Monitor")
    print("Optimized for 100kHz+ data collection")
    print("Features: Real-time visualization, FFT analysis, Smart downsampling")
    print("Using PyQt5 + PyQtGraph for high-performance plotting")
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = HighFreqMonitorGUI()
    window.run()
    
    sys.exit(app.exec_()) 