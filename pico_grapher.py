import sys
import serial
import serial.tools.list_ports
from collections import deque
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                            QHBoxLayout, QWidget, QPushButton, QComboBox,
                            QLabel, QTextEdit)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
import pyqtgraph as pg

class SerialReaderThread(QThread):
    """Thread to read serial data from the Pi Pico without blocking the UI."""
    batch_received = pyqtSignal(list)  # Emits a list of integer ADC values
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_connected = False
        self.should_run = True
        self.port_name = ""
        self.baudrate = 115200  # Standard for Pico USB CDC

    def connect_serial(self, port):
        self.port_name = port
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            self.serial_port = serial.Serial(port, self.baudrate, timeout=1)
            self.is_connected = True
            self.status_changed.emit(f"Connected to {port}")
            return True
        except Exception as e:
            error_msg = f"Failed to connect to {port}: {e}"
            self.error_occurred.emit(error_msg)
            self.is_connected = False
            return False

    def disconnect(self):
        self.should_run = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.status_changed.emit("Disconnected")

    def run(self):
        reading_batch = False
        current_batch = []
        while self.should_run and self.is_connected:
            try:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    if line.startswith('BATCH,'):
                        reading_batch = True
                        current_batch = []
                    elif line == 'END_BATCH':
                        if reading_batch:
                            self.batch_received.emit(current_batch)
                            reading_batch = False
                    elif reading_batch:
                        try:
                            current_batch.append(int(line))
                        except ValueError:
                            pass # Ignore non-integer lines in a batch
                    elif "READY" in line:
                        self.status_changed.emit("Pico is READY. Receiving data...")

            except Exception as e:
                self.error_occurred.emit(f"Serial reading error: {e}")
                self.disconnect()
                break

class PicoGrapherApp(QMainWindow):
    """Main application window for graphing Pico data."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pi Pico Real-time ADC Graph")
        self.setGeometry(100, 100, 1200, 800)

        self.max_points = 30000
        self.data_buffer = deque(maxlen=self.max_points)
        
        self.serial_thread = SerialReaderThread()
        self.serial_thread.batch_received.connect(self.update_data)
        self.serial_thread.error_occurred.connect(self.log_debug)
        self.serial_thread.status_changed.connect(self.update_status)
        
        self.setup_ui()
        self.refresh_ports()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Pico Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(250)
        control_layout.addWidget(self.port_combo)
        
        self.refresh_btn = QPushButton("Refresh Ports")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        control_layout.addWidget(self.refresh_btn)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_btn)

        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self.clear_data)
        control_layout.addWidget(self.clear_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.status_label = QLabel("Disconnected")
        layout.addWidget(self.status_label)
        
        self.plot = self.graph_widget.addPlot(title="Pico ADC Voltage (GPIO26)")
        self.plot.setLabel('left', 'Voltage (V)')
        self.plot.setLabel('bottom', 'Sample Number')
        self.plot.showGrid(x=True, y=True)
        self.plot.setYRange(0, 3.3, padding=0.1)
        self.curve = self.plot.plot(pen='c')
        self.plot.setDownsampling(mode='peak')
        self.plot.setClipToView(True)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [p for p in serial.tools.list_ports.comports() if "serial" in p.description.lower() or "pico" in p.description.lower() or "CP210x" in p.description]
        if not ports:
            ports = serial.tools.list_ports.comports()
        
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")

    def toggle_connection(self):
        if not self.serial_thread.isRunning():
            port_text = self.port_combo.currentText()
            if not port_text:
                self.log_debug("No port selected!")
                return
            port = port_text.split(' - ')[0]
            if self.serial_thread.connect_serial(port):
                self.serial_thread.start()
                self.connect_btn.setText("Disconnect")
        else:
            self.serial_thread.disconnect()
            self.serial_thread.wait()
            self.connect_btn.setText("Connect")

    def clear_data(self):
        self.data_buffer.clear()
        self.update_plot()

    def update_data(self, batch):
        # Convert raw 16-bit ADC values to voltage
        voltage_batch = (np.array(batch, dtype=np.float32) / 65535.0) * 3.3
        self.data_buffer.extend(voltage_batch)
        self.update_plot()

    def update_plot(self):
        y_data = np.array(self.data_buffer)
        x_data = np.arange(len(y_data))
        
        self.curve.setData(y=y_data, x=x_data)
        
        if len(x_data) > 0:
            window_size = min(len(x_data), 5000)
            start = max(0, len(x_data) - window_size)
            self.plot.setXRange(start, start + window_size, padding=0)

    def log_debug(self, message):
        print(f"LOG: {message}")
    
    def update_status(self, status):
        self.status_label.setText(status)
        self.log_debug(status)

    def closeEvent(self, event):
        self.serial_thread.disconnect()
        self.serial_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = PicoGrapherApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 