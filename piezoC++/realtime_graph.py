import sys
import serial
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
from collections import deque

class PicoDataReader(QThread):
    data_received = pyqtSignal(list, int, int)  # Signal: (voltage_data, batch_number, sample_count)
    status_update = pyqtSignal(str)  # Signal: status message
    error_occurred = pyqtSignal(str)  # Signal: error message
    
    def __init__(self, com_port):
        super().__init__()
        self.com_port = com_port
        self.running = False
        self.serial_port = None
        
    def run(self):
        try:
            print(f"[PYTHON] Connecting to {self.com_port}...")
            self.status_update.emit(f"Connecting to {self.com_port}...")
            self.serial_port = serial.Serial(self.com_port, 115200, timeout=1)
            print(f"[PYTHON] Connected to {self.com_port}")
            self.status_update.emit(f"Connected to {self.com_port}")
            self.running = True
            
            current_batch = []
            in_batch = False
            batch_number = 0
            expected_samples = 0
            
            while self.running:
                try:
                    line = self.serial_port.readline().decode().strip()
                    if not line:
                        continue
                        
                    if line.startswith("BATCH_START:"):
                        parts = line.split(":")
                        batch_number = int(parts[1])
                        expected_samples = int(parts[2])
                        current_batch = []
                        in_batch = True
                        
                    elif line.startswith("BATCH_END:"):
                        batch_num = int(line.split(":")[1])
                        in_batch = False
                        
                        if current_batch and len(current_batch) > 0:
                            self.data_received.emit(current_batch.copy(), batch_num, len(current_batch))
                        
                    elif in_batch:
                        try:
                            voltage = float(line)
                            current_batch.append(voltage)
                        except ValueError:
                            pass  # Skip invalid lines
                            
                    elif line.startswith("===") or line.startswith("Status:") or line.startswith("Config:"):
                        print(f"[PICO] {line}")
                        self.status_update.emit(line)
                        
                except serial.SerialTimeoutException:
                    continue
                except Exception as e:
                    print(f"[PYTHON] Read error: {e}")
                    self.error_occurred.emit(f"Read error: {e}")
                    break
                    
        except serial.SerialException as e:
            print(f"[PYTHON] Serial connection error: {e}")
            self.error_occurred.emit(f"Serial connection error: {e}")
        except Exception as e:
            print(f"[PYTHON] Unexpected error: {e}")
            self.error_occurred.emit(f"Unexpected error: {e}")
        finally:
            if self.serial_port:
                self.serial_port.close()
                print(f"[PYTHON] Disconnected from {self.com_port}")
    
    def stop(self):
        self.running = False
        self.wait()

class RealtimeGraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pico ADC Real-time Graph (30kHz Fast)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage - optimized for fast updates
        self.voltage_history = deque(maxlen=90000)  # Keep last 3 seconds (30 batches * 3k samples)
        self.time_history = deque(maxlen=90000)
        self.batch_count = 0
        self.total_samples = 0
        self.last_update_time = time.time()
        
        # Setup UI
        self.setup_ui()
        
        # Data reader thread
        self.data_reader = None
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = QHBoxLayout()
        
        self.port_combo = QComboBox()
        self.refresh_ports()
        control_panel.addWidget(QLabel("COM Port:"))
        control_panel.addWidget(self.port_combo)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        control_panel.addWidget(self.connect_btn)
        
        self.refresh_btn = QPushButton("Refresh Ports")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        control_panel.addWidget(self.refresh_btn)
        
        control_panel.addStretch()
        
        # Status labels
        self.status_label = QLabel("Status: Ready")
        self.batch_label = QLabel("Batches: 0")
        self.sample_label = QLabel("Samples: 0")
        self.rate_label = QLabel("Rate: 0 Hz")
        
        control_panel.addWidget(self.status_label)
        control_panel.addWidget(self.batch_label)
        control_panel.addWidget(self.sample_label)
        control_panel.addWidget(self.rate_label)
        
        layout.addLayout(control_panel)
        
        # Graph - optimized settings
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setLabel('left', 'Voltage', units='V')
        self.graph_widget.setLabel('bottom', 'Time', units='s')
        self.graph_widget.setTitle('Pico ADC Real-time Data (30kHz - Fast Mode)')
        self.graph_widget.showGrid(x=True, y=True)
        
        # Plot line - optimized for speed
        self.plot_line = self.graph_widget.plot([], [], pen=pg.mkPen(color='cyan', width=1))
        
        layout.addWidget(self.graph_widget)
        
        # Info panel
        info_layout = QHBoxLayout()
        self.info_label = QLabel("Fast mode: 3k samples every 0.1s")
        info_layout.addWidget(self.info_label)
        layout.addLayout(info_layout)
        
    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}", port.device)
            
    def toggle_connection(self):
        if self.data_reader and self.data_reader.running:
            # Disconnect
            print("[PYTHON] Disconnecting...")
            self.data_reader.stop()
            self.data_reader = None
            self.connect_btn.setText("Connect")
            self.status_label.setText("Status: Disconnected")
        else:
            # Connect
            if self.port_combo.count() == 0:
                QMessageBox.warning(self, "No Ports", "No COM ports available!")
                return
                
            com_port = self.port_combo.currentData()
            print(f"[PYTHON] Connecting to {com_port}...")
            self.data_reader = PicoDataReader(com_port)
            self.data_reader.data_received.connect(self.on_data_received)
            self.data_reader.status_update.connect(self.on_status_update)
            self.data_reader.error_occurred.connect(self.on_error)
            self.data_reader.start()
            
            self.connect_btn.setText("Disconnect")
            
    def on_data_received(self, voltage_data, batch_number, sample_count):
        # Calculate timing
        current_time = time.time()
        time_since_last = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Add new data to history
        start_time = self.total_samples / 30000.0  # Convert sample count to seconds (30kHz)
        time_points = np.linspace(start_time, start_time + len(voltage_data)/30000.0, len(voltage_data))
        
        self.voltage_history.extend(voltage_data)
        self.time_history.extend(time_points)
        
        self.batch_count += 1
        self.total_samples += len(voltage_data)
        
        # Update plot (but not too frequently to avoid lag)
        if self.batch_count % 3 == 0:  # Update plot every 3 batches (0.3 seconds)
            self.update_plot()
        
        # Update labels
        self.batch_label.setText(f"Batches: {self.batch_count}")
        self.sample_label.setText(f"Samples: {self.total_samples:,}")
        
        # Calculate sample rate
        if time_since_last > 0:
            rate = len(voltage_data) / time_since_last
            self.rate_label.setText(f"Rate: {rate:.0f} Hz")
        
        # Print status every 10 batches
        if self.batch_count % 10 == 0:
            print(f"[PYTHON] Batch {batch_number}: {sample_count} samples, total: {self.total_samples:,}")
        
    def update_plot(self):
        if len(self.voltage_history) > 0:
            # Convert to numpy arrays for plotting
            voltages = np.array(list(self.voltage_history))
            times = np.array(list(self.time_history))
            
            # Update plot
            self.plot_line.setData(times, voltages)
            
            # Auto-scale to show recent data
            if len(times) > 0:
                self.graph_widget.setXRange(max(0, times[-1] - 3), times[-1] + 0.1)  # Show last 3 seconds
                
    def on_status_update(self, message):
        self.status_label.setText(f"Status: {message}")
        
    def on_error(self, error_message):
        print(f"[PYTHON] Error: {error_message}")
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Error", error_message)
        if self.data_reader:
            self.data_reader.stop()
            self.data_reader = None
            self.connect_btn.setText("Connect")

def main():
    print("[PYTHON] Starting Pico ADC Fast Real-time Graph (30kHz)")
    
    app = QApplication(sys.argv)
    
    # Check if PyQt and pyqtgraph are available
    try:
        window = RealtimeGraphWindow()
        window.show()
        print("[PYTHON] Fast mode ready - 3k samples every 0.1s")
        sys.exit(app.exec_())
    except ImportError as e:
        print(f"[PYTHON] Missing required package: {e}")
        print("[PYTHON] Please install: pip install PyQt5 pyqtgraph numpy")
        return

if __name__ == "__main__":
    main() 