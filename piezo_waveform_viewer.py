#!/usr/bin/env python3
"""
Piezo Sensor Waveform Viewer - High Performance Edition

Real-time data acquisition and visualization for Teensy 4.0 piezo sensor system.
Reads 15kHz ADC data from COM5, converts to voltage, and displays waveform.

Requirements:
- pyserial
- PyQt5
- pyqtgraph
- numpy

Hardware:
- Teensy 4.0 with piezo sensor on A0
- 20x opamp circuit
- USB connection to COM5

Author: Generated for leak detection research
"""

import sys
import struct
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import serial
import time

class PiezoWaveformViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuration
        self.SERIAL_PORT = 'COM5'
        self.BAUD_RATE = 500000
        self.SAMPLE_RATE = 15000  # Hz
        self.BUFFER_SIZE = 45000  # 3 seconds at 15kHz
        self.ADC_MAX = 4095  # 12-bit ADC
        self.VOLTAGE_MAX = 3.3  # Volts
        
        # Performance optimizations
        self.update_rate_samples = 50  # Update plot every 50 samples (~3ms intervals) - FASTER!
        self.plot_downsample = 2  # Plot every Nth point for performance
        
        # Data storage - using numpy arrays for better performance
        self.voltage_buffer = np.zeros(self.BUFFER_SIZE, dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        self.current_sample = 0
        
        # Serial connection
        self.serial_connection = None
        self.is_connected = False
        
        # Timing
        self.last_update_time = time.time()
        self.samples_received = 0
        
        # Initialize UI
        self.init_ui()
        self.init_plot()
        
        # Setup timer for data acquisition - faster timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Piezo Sensor Waveform Viewer - High Performance')
        self.setGeometry(100, 100, 1400, 900)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: white; }
            QLabel { color: white; }
            QPushButton { 
                background-color: #404040; 
                color: white; 
                border: 1px solid #606060;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #353535; }
            QComboBox { 
                background-color: #404040; 
                color: white; 
                border: 1px solid #606060;
                padding: 3px;
            }
        """)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = QtWidgets.QHBoxLayout()
        
        # Connection controls
        self.connect_button = QtWidgets.QPushButton('Connect')
        self.connect_button.clicked.connect(self.toggle_connection)
        self.connect_button.setFixedSize(100, 35)
        
        self.status_label = QtWidgets.QLabel('Disconnected')
        self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold; font-size: 12px;")
        
        # Statistics
        self.stats_label = QtWidgets.QLabel('Samples: 0 | Rate: 0 Hz | Voltage: 0.00V')
        self.stats_label.setStyleSheet("font-family: 'Courier New', monospace; color: #4ecdc4; font-size: 11px;")
        
        # Port selection
        port_label = QtWidgets.QLabel('Port:')
        port_label.setStyleSheet("color: white; font-weight: bold;")
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.addItem('COM5')
        self.port_combo.setEditable(True)
        
        control_panel.addWidget(self.connect_button)
        control_panel.addWidget(self.status_label)
        control_panel.addWidget(port_label)
        control_panel.addWidget(self.port_combo)
        control_panel.addStretch()
        control_panel.addWidget(self.stats_label)
        
        layout.addLayout(control_panel)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
    def init_plot(self):
        """Initialize the plot configuration with black background and white lines"""
        # Set black background
        self.plot_widget.setBackground('k')  # Black background
        
        # Configure labels and title with white text
        label_style = {'color': 'white', 'font-size': '12pt'}
        self.plot_widget.setLabel('left', 'Voltage (V)', **label_style)
        self.plot_widget.setLabel('bottom', 'Sample Index', **label_style)
        self.plot_widget.setTitle('Piezo Sensor Waveform (15kHz Real-time)', color='white', size='14pt')
        
        # Set plot range
        self.plot_widget.setYRange(0, self.VOLTAGE_MAX)
        self.plot_widget.setXRange(0, self.BUFFER_SIZE)
        
        # Configure plot appearance with white grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.4)
        self.plot_widget.getPlotItem().getAxis('left').setPen(color='white')
        self.plot_widget.getPlotItem().getAxis('bottom').setPen(color='white')
        self.plot_widget.getPlotItem().getAxis('left').setTextPen(color='white')
        self.plot_widget.getPlotItem().getAxis('bottom').setTextPen(color='white')
        
        # Create plot curve with white line
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='white', width=1.5))
        
        # Add voltage reference lines in different colors
        self.plot_widget.addLine(y=0, pen=pg.mkPen('#ff6b6b', style=QtCore.Qt.DashLine, width=1))  # Red for 0V
        self.plot_widget.addLine(y=self.VOLTAGE_MAX, pen=pg.mkPen('#4ecdc4', style=QtCore.Qt.DashLine, width=1))  # Cyan for 3.3V
        self.plot_widget.addLine(y=self.VOLTAGE_MAX/2, pen=pg.mkPen('#45b7d1', style=QtCore.Qt.DashLine, width=1))  # Blue for 1.65V
        
        # Performance optimizations
        self.plot_widget.setClipToView(True)
        self.plot_widget.setDownsampling(mode='peak')
        self.plot_widget.setRange(xRange=[0, self.BUFFER_SIZE], yRange=[0, self.VOLTAGE_MAX])
        
    def toggle_connection(self):
        """Toggle serial connection"""
        if self.is_connected:
            self.disconnect_serial()
        else:
            self.connect_serial()
    
    def connect_serial(self):
        """Establish serial connection"""
        try:
            port = self.port_combo.currentText()
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=self.BAUD_RATE,
                timeout=0.01,  # Reduced timeout for faster response
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Clear any existing data
            self.serial_connection.flushInput()
            
            self.is_connected = True
            self.connect_button.setText('Disconnect')
            self.status_label.setText('Connected')
            self.status_label.setStyleSheet("color: #51cf66; font-weight: bold; font-size: 12px;")
            
            # Start data acquisition timer - much faster!
            self.timer.start(0)  # 0ms timer for maximum speed
            
            print(f"Connected to {port} at {self.BAUD_RATE} baud")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                'Connection Error', 
                f'Failed to connect to {self.port_combo.currentText()}:\n{str(e)}'
            )
    
    def disconnect_serial(self):
        """Close serial connection"""
        self.timer.stop()
        
        if self.serial_connection:
            self.serial_connection.close()
            self.serial_connection = None
        
        self.is_connected = False
        self.connect_button.setText('Connect')
        self.status_label.setText('Disconnected')
        self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold; font-size: 12px;")
        
        print("Disconnected from serial port")
    
    def adc_to_voltage(self, adc_value):
        """Convert 12-bit ADC value to voltage"""
        return (adc_value / self.ADC_MAX) * self.VOLTAGE_MAX
    
    def update_data(self):
        """Read serial data and update plot - optimized for speed"""
        if not self.is_connected or not self.serial_connection:
            return
        
        try:
            # Read available data
            bytes_available = self.serial_connection.in_waiting
            
            # Process larger chunks for better performance
            if bytes_available >= 20:  # Process at least 10 samples at once
                num_samples = min(bytes_available // 2, 500)  # Cap at 500 samples per update
                
                # Read binary data
                data = self.serial_connection.read(num_samples * 2)
                
                # Process all samples in batch using numpy for speed
                if len(data) >= 2:
                    # Convert bytes to uint16 array
                    adc_values = np.frombuffer(data, dtype=np.uint16)
                    
                    # Convert to voltages
                    voltages = (adc_values.astype(np.float32) / self.ADC_MAX) * self.VOLTAGE_MAX
                    
                    # Add to circular buffer
                    for voltage in voltages:
                        self.voltage_buffer[self.buffer_index] = voltage
                        self.buffer_index = (self.buffer_index + 1) % self.BUFFER_SIZE
                        if self.buffer_index == 0:
                            self.buffer_full = True
                        self.current_sample += 1
                        self.samples_received += 1
                    
                    # Update plot more frequently
                    if self.samples_received % self.update_rate_samples == 0:
                        self.update_plot()
                        self.update_statistics()
        
        except Exception as e:
            print(f"Data acquisition error: {e}")
            self.disconnect_serial()
            
    def update_plot(self):
        """Update the waveform plot - optimized for performance"""
        if self.current_sample > 0:
            if self.buffer_full:
                # Use the entire buffer, reordered to show chronological data
                plot_data = np.concatenate([
                    self.voltage_buffer[self.buffer_index:],
                    self.voltage_buffer[:self.buffer_index]
                ])
                x_data = np.arange(self.current_sample - self.BUFFER_SIZE, self.current_sample)
            else:
                # Use only filled portion of buffer
                plot_data = self.voltage_buffer[:self.buffer_index]
                x_data = np.arange(self.buffer_index)
            
            # Downsample for performance if needed
            if len(plot_data) > 10000:  # Only downsample if we have lots of data
                step = max(1, len(plot_data) // 10000)
                plot_data = plot_data[::step]
                x_data = x_data[::step]
            
            # Update curve data
            self.curve.setData(x_data, plot_data)
            
            # Auto-range X axis to show latest data
            if self.buffer_full:
                self.plot_widget.setXRange(
                    self.current_sample - self.BUFFER_SIZE, 
                    self.current_sample,
                    padding=0
                )
    
    def update_statistics(self):
        """Update statistics display"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed > 0:
            sample_rate = self.samples_received / elapsed if elapsed > 0 else 0
            
            # Get current voltage
            current_voltage = self.voltage_buffer[(self.buffer_index - 1) % self.BUFFER_SIZE] if self.current_sample > 0 else 0
            
            # Update display with more info
            buffer_fill = self.buffer_index if not self.buffer_full else self.BUFFER_SIZE
            self.stats_label.setText(
                f'Samples: {self.samples_received:,} | '
                f'Rate: {sample_rate:.0f} Hz | '
                f'Voltage: {current_voltage:.3f}V | '
                f'Buffer: {buffer_fill:,}/{self.BUFFER_SIZE:,} | '
                f'Update: {1000/max(elapsed*1000, 1):.1f} Hz'
            )
            
            # Reset counters periodically
            if elapsed > 2.0:  # Reset every 2 seconds for more responsive display
                self.last_update_time = current_time
                self.samples_received = 0
    
    def closeEvent(self, event):
        """Handle application close"""
        self.disconnect_serial()
        event.accept()

def main():
    """Main application entry point"""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Piezo Waveform Viewer - High Performance')
    
    # Enable high DPI scaling
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    
    # Create and show main window
    viewer = PiezoWaveformViewer()
    viewer.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 