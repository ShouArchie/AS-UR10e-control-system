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
        self.SERIAL_PORT = 'COM7'
        self.BAUD_RATE = 921600  # Increased for 20kHz data rate (20k samples/sec * 2 bytes = 40KB/s)
        self.SAMPLE_RATE = 20000  # Hz
        self.BUFFER_SIZE = 60000  # 3 seconds at 20kHz
        self.ADC_MAX = 4095  # 12-bit ADC
        self.VOLTAGE_MAX = 3.3  # Opamp powered from 5V, adjust if using voltage divider
        
        # Performance optimizations - 20kHz OPTIMIZED  
        self.update_rate_samples = 10   # Update plot every 10 samples for 20kHz
        self.plot_downsample = 2  # Plot every Nth point for performance
        self.min_bytes_to_process = 2  # Process any available data (minimum 1 sample)
        
        # Data storage - dual channel ADC values
        self.adc_buffer_A0 = np.zeros(self.BUFFER_SIZE, dtype=np.uint16)  # A0 raw ADC values 0-4095
        self.adc_buffer_A1 = np.zeros(self.BUFFER_SIZE, dtype=np.uint16)  # A1 raw ADC values 0-4095
        self.buffer_index = 0
        self.buffer_full = False
        self.current_sample = 0
        
        # ADC tracking for diagnostics (dual channel)
        self.max_adc_seen_A0 = 0
        self.min_adc_seen_A0 = 4095
        self.max_adc_seen_A1 = 0
        self.min_adc_seen_A1 = 4095
        
        # Software high-pass filter option
        self.enable_high_pass = False  # Set to True to remove DC component
        self.dc_baseline = None
        
        # Auto-ranging to use full ADC range
        self.enable_auto_range = False  # Set to True to maximize ADC utilization
        self.signal_min = None
        self.signal_max = None
        self.auto_range_samples = 0
        
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
        self.setWindowTitle('Piezo Sensor Waveform Viewer - 20kHz High Performance')
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
        self.stats_label = QtWidgets.QLabel('Samples: 0 | Rate: 0 Hz | ADC: 0')
        self.stats_label.setStyleSheet("font-family: 'Courier New', monospace; color: #4ecdc4; font-size: 11px;")
        
        # Port selection
        port_label = QtWidgets.QLabel('Port:')
        port_label.setStyleSheet("color: white; font-weight: bold;")
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.addItem('COM7')
        self.port_combo.setEditable(True)
        
        # Auto-range control
        self.auto_range_check = QtWidgets.QCheckBox('Auto-Range')
        self.auto_range_check.setChecked(self.enable_auto_range)
        self.auto_range_check.toggled.connect(self.toggle_auto_range)
        self.auto_range_check.setStyleSheet("color: white;")
        
        control_panel.addWidget(self.connect_button)
        control_panel.addWidget(self.status_label)
        control_panel.addWidget(port_label)
        control_panel.addWidget(self.port_combo)
        control_panel.addWidget(self.auto_range_check)
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
        self.plot_widget.setLabel('left', 'ADC Value (0-4095)', **label_style)
        self.plot_widget.setLabel('bottom', 'Sample Index', **label_style)
        self.plot_widget.setTitle('Dual Piezo Sensors A0/A1 Raw ADC Values (20kHz Real-time)', color='white', size='14pt')
        
        # Set consistent plot range for ADC values
        self.plot_widget.setYRange(300, 4500)  # Consistent view range
        self.plot_widget.setXRange(0, self.BUFFER_SIZE)
        
        # Configure plot appearance with white grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.4)
        self.plot_widget.getPlotItem().getAxis('left').setPen(color='white')
        self.plot_widget.getPlotItem().getAxis('bottom').setPen(color='white')
        self.plot_widget.getPlotItem().getAxis('left').setTextPen(color='white')
        self.plot_widget.getPlotItem().getAxis('bottom').setTextPen(color='white')
        
        # Create plot curves for dual channels
        self.curve_A0 = self.plot_widget.plot(pen=pg.mkPen(color='white', width=1.5))  # A0 in white
        self.curve_A1 = self.plot_widget.plot(pen=pg.mkPen(color='lightblue', width=1.5))  # A1 in light blue
        
        # Add ADC reference lines optimized for 300-4500 range
        self.plot_widget.addLine(y=500, pen=pg.mkPen('#ff6b6b', style=QtCore.Qt.DashLine, width=1))  # Red for low baseline
        self.plot_widget.addLine(y=4095, pen=pg.mkPen('#4ecdc4', style=QtCore.Qt.DashLine, width=1))  # Cyan for max ADC (4095)
        self.plot_widget.addLine(y=2048, pen=pg.mkPen('#45b7d1', style=QtCore.Qt.DashLine, width=1))  # Blue for midpoint (2048)
        self.plot_widget.addLine(y=1000, pen=pg.mkPen('#ffa500', style=QtCore.Qt.DashLine, width=1))  # Orange for low reference
        self.plot_widget.addLine(y=3000, pen=pg.mkPen('#9370db', style=QtCore.Qt.DashLine, width=1))  # Purple for high reference
        
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
            print("Waiting for data from Teensy...")
            
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
            
            # Debug: print bytes available occasionally
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 10000 == 0 and bytes_available > 0:
                print(f"Debug: {bytes_available} bytes available")
            
            # Process any available data
            if bytes_available >= self.min_bytes_to_process:
                # Read all available data
                data = self.serial_connection.read(bytes_available)
                
                # Process data in pairs (each sample is 2 bytes)
                if len(data) >= 2:
                    # Check for text data (common cause of spikes)
                    # Look for printable ASCII characters that indicate text mixed with binary
                    text_chars = sum(1 for b in data[:20] if 32 <= b <= 126)  # Check first 20 bytes
                    if text_chars > 10:  # If more than half are text characters
                        print(f"Warning: Text data detected, skipping: {data[:50]}")
                        return
                    
                    # Ensure we have complete samples (even number of bytes)
                    complete_bytes = (len(data) // 2) * 2
                    if complete_bytes > 0:
                        # Convert bytes to uint16 array (little-endian)
                        try:
                            adc_values = np.frombuffer(data[:complete_bytes], dtype='<u2')  # little-endian uint16
                            
                            # Filter out invalid ADC values (spikes)
                            valid_mask = (adc_values <= self.ADC_MAX) & (adc_values >= 0)
                            invalid_count = np.sum(~valid_mask)
                            
                            # Debug: Check filtering
                            if self._debug_counter % 1000 == 0:
                                print(f"Filter Debug: Total values: {len(adc_values)}, Invalid: {invalid_count}, Valid: {np.sum(valid_mask)}")
                            
                            if invalid_count > 0:
                                print(f"Warning: {invalid_count} invalid ADC values detected (spikes)")
                                print(f"Invalid values: {adc_values[~valid_mask][:10]}")  # Show first 10 invalid values
                                # Replace invalid values with previous valid value or median
                                adc_values = adc_values[valid_mask]
                            
                            if len(adc_values) == 0:
                                print("Error: No valid ADC data after filtering!")
                                return  # Skip this batch if no valid data
                            
                            # Debug: print first few values occasionally
                            if self._debug_counter % 1000 == 0:  # More frequent debugging
                                print(f"Debug: Got {len(adc_values)} valid samples, first 10 ADC values: {adc_values[:10]}")
                                print(f"Debug: ADC min/max: {adc_values.min()}/{adc_values.max()} (max possible: {self.ADC_MAX})")
                                # Check if data looks like it should be interleaved
                                if len(adc_values) >= 10:
                                    print(f"Debug: Even indices (should be A0): {adc_values[0::2][:5]}")
                                    print(f"Debug: Odd indices (should be A1): {adc_values[1::2][:5]}")
                                    print(f"Debug: Raw bytes: {[hex(b) for b in data[:20]]}")
                            
                            # Track ADC extremes for diagnostics (dual channel)
                            # Note: adc_values contains interleaved A0,A1,A0,A1... data
                            if len(adc_values) >= 2:
                                a0_values = adc_values[0::2]  # Every other value starting from 0
                                a1_values = adc_values[1::2]  # Every other value starting from 1
                                
                                if len(a0_values) > 0:
                                    current_max_a0 = a0_values.max()
                                    current_min_a0 = a0_values.min()
                                    if current_max_a0 > self.max_adc_seen_A0:
                                        self.max_adc_seen_A0 = current_max_a0
                                    if current_min_a0 < self.min_adc_seen_A0:
                                        self.min_adc_seen_A0 = current_min_a0
                                
                                if len(a1_values) > 0:
                                    current_max_a1 = a1_values.max()
                                    current_min_a1 = a1_values.min()
                                    if current_max_a1 > self.max_adc_seen_A1:
                                        self.max_adc_seen_A1 = current_max_a1
                                    if current_min_a1 < self.min_adc_seen_A1:
                                        self.min_adc_seen_A1 = current_min_a1
                            
                            # Store raw ADC values directly (dual channel processing)
                            raw_adc_values = adc_values.astype(np.uint16)
                            
                            # Split into A0 and A1 channels (interleaved data: A0,A1,A0,A1...)
                            if len(raw_adc_values) >= 2:
                                a0_values = raw_adc_values[0::2]  # A0 values
                                a1_values = raw_adc_values[1::2]  # A1 values
                                
                                # Optional software high-pass filter (removes DC baseline) - per channel
                                if self.enable_high_pass:
                                    # Handle A0 baseline
                                    if not hasattr(self, 'dc_baseline_A0') or self.dc_baseline_A0 is None:
                                        self.dc_baseline_A0 = np.mean(a0_values)
                                    else:
                                        self.dc_baseline_A0 = 0.999 * self.dc_baseline_A0 + 0.001 * np.mean(a0_values)
                                    a0_values = a0_values - self.dc_baseline_A0
                                    
                                    # Handle A1 baseline
                                    if not hasattr(self, 'dc_baseline_A1') or self.dc_baseline_A1 is None:
                                        self.dc_baseline_A1 = np.mean(a1_values)
                                    else:
                                        self.dc_baseline_A1 = 0.999 * self.dc_baseline_A1 + 0.001 * np.mean(a1_values)
                                    a1_values = a1_values - self.dc_baseline_A1
                                
                                # Additional spike filter on ADC level per channel
                                a0_spikes = (a0_values > self.ADC_MAX) | (a0_values < 0)
                                a1_spikes = (a1_values > self.ADC_MAX) | (a1_values < 0)
                                
                                if np.any(a0_spikes):
                                    spike_count = np.sum(a0_spikes)
                                    print(f"Warning: {spike_count} A0 ADC spikes detected and filtered")
                                    a0_values = a0_values[~a0_spikes]
                                
                                if np.any(a1_spikes):
                                    spike_count = np.sum(a1_spikes)
                                    print(f"Warning: {spike_count} A1 ADC spikes detected and filtered")
                                    a1_values = a1_values[~a1_spikes]
                                
                                # Debug: print ADC range per channel
                                if self._debug_counter % 1000 == 0:
                                    if len(a0_values) > 0:
                                        print(f"Debug: A0 - Count: {len(a0_values)}, Range: {a0_values.min()} to {a0_values.max()}, First 5: {a0_values[:5]}")
                                    if len(a1_values) > 0:
                                        print(f"Debug: A1 - Count: {len(a1_values)}, Range: {a1_values.min()} to {a1_values.max()}, First 5: {a1_values[:5]}")
                                    
                                    # Check if A0 and A1 are too similar (indicating parsing error)
                                    if len(a0_values) > 0 and len(a1_values) > 0:
                                        a0_mean = np.mean(a0_values)
                                        a1_mean = np.mean(a1_values)
                                        diff = abs(a0_mean - a1_mean)
                                        print(f"Debug: Channel difference - A0 mean: {a0_mean:.1f}, A1 mean: {a1_mean:.1f}, Diff: {diff:.1f}")
                                        if diff < 10:  # If channels are too similar, there might be a parsing issue
                                            print(f"WARNING: Channels are very similar! Possible data parsing issue.")
                                
                                # Add to circular buffers for both channels
                                for a0_val, a1_val in zip(a0_values, a1_values):
                                    self.adc_buffer_A0[self.buffer_index] = int(a0_val)
                                    self.adc_buffer_A1[self.buffer_index] = int(a1_val)
                                    self.buffer_index = (self.buffer_index + 1) % self.BUFFER_SIZE
                                    if self.buffer_index == 0:
                                        self.buffer_full = True
                                    self.current_sample += 1
                                    self.samples_received += 1
                            
                                # Debug: Check if data is being stored
                                if self._debug_counter % 1000 == 0:
                                    print(f"Debug: Stored {min(len(a0_values), len(a1_values))} dual-channel ADC values, buffer_index now: {self.buffer_index}")
                                    print(f"Debug: Last few A0 buffer values: {self.adc_buffer_A0[max(0, self.buffer_index-3):self.buffer_index]}")
                                    print(f"Debug: Last few A1 buffer values: {self.adc_buffer_A1[max(0, self.buffer_index-3):self.buffer_index]}")
                                    print(f"Debug: Current sample count: {self.current_sample}, buffer_full: {self.buffer_full}")
                            
                            # Update plot MUCH more frequently - after every batch!
                            self.update_plot()
                            # Force GUI update for responsiveness
                            QtWidgets.QApplication.processEvents()
                            
                            # Debug: Confirm update was called
                            if self._debug_counter % 1000 == 0:
                                print(f"Debug: update_plot() and processEvents() called, samples: {self.samples_received}")
                            
                            if self.samples_received % 100 == 0:  # Update stats less frequently
                                self.update_statistics()
                                
                        except Exception as e:
                            print(f"Data conversion error: {e}, data length: {len(data)}")
                            print(f"First 10 bytes: {[hex(b) for b in data[:10]]}")
                            # Skip this data and continue
        
        except Exception as e:
            print(f"Data acquisition error: {e}")
            self.disconnect_serial()
            
    def update_plot(self):
        """Update the waveform plot - optimized for performance (dual channel raw ADC values)"""
        if self.current_sample > 0:
            if self.buffer_full:
                # Use the entire buffer, reordered to show chronological data
                plot_data_a0 = np.concatenate([
                    self.adc_buffer_A0[self.buffer_index:],
                    self.adc_buffer_A0[:self.buffer_index]
                ])
                plot_data_a1 = np.concatenate([
                    self.adc_buffer_A1[self.buffer_index:],
                    self.adc_buffer_A1[:self.buffer_index]
                ])
                x_data = np.arange(self.current_sample - self.BUFFER_SIZE, self.current_sample)
            else:
                # Use only filled portion of buffer
                plot_data_a0 = self.adc_buffer_A0[:self.buffer_index]
                plot_data_a1 = self.adc_buffer_A1[:self.buffer_index]
                x_data = np.arange(self.buffer_index)
            
            # Debug: Check if plot data is valid
            if hasattr(self, '_plot_debug_counter'):
                self._plot_debug_counter += 1
            else:
                self._plot_debug_counter = 0
                
            if self._plot_debug_counter % 500 == 0:
                print(f"Plot Debug: A0 data length: {len(plot_data_a0)}, range: {plot_data_a0.min()}-{plot_data_a0.max()}")
                print(f"Plot Debug: A1 data length: {len(plot_data_a1)}, range: {plot_data_a1.min()}-{plot_data_a1.max()}")
                print(f"Plot Debug: x_data length: {len(x_data)}, range: {x_data.min()}-{x_data.max()}")
            
            # Aggressive downsampling for ultra-fast updates
            if len(plot_data_a0) > 5000:  # Downsample sooner for speed
                step = max(1, len(plot_data_a0) // 5000)
                plot_data_a0 = plot_data_a0[::step]
                plot_data_a1 = plot_data_a1[::step]
                x_data = x_data[::step]
            
            # Update curve data for both channels
            self.curve_A0.setData(x_data, plot_data_a0)
            self.curve_A1.setData(x_data, plot_data_a1)
            
            # Auto-range both X and Y axis to show data properly
            if self.buffer_full:
                self.plot_widget.setXRange(
                    self.current_sample - self.BUFFER_SIZE, 
                    self.current_sample,
                    padding=0
                )
            else:
                # Update X range for growing buffer
                self.plot_widget.setXRange(0, max(1000, self.buffer_index), padding=0)
            
            # Set consistent Y-axis range for ADC values
            self.plot_widget.setYRange(300, 4500, padding=0)
        else:
            # Debug: No samples to plot
            if hasattr(self, '_no_sample_debug'):
                self._no_sample_debug += 1
            else:
                self._no_sample_debug = 0
            
            if self._no_sample_debug % 1000 == 0:
                print(f"Plot Debug: No samples to plot (current_sample: {self.current_sample})")
    
    def update_statistics(self):
        """Update statistics display"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed > 0:
            sample_rate = self.samples_received / elapsed if elapsed > 0 else 0
            
            # Get current ADC values for both channels
            current_adc_a0 = self.adc_buffer_A0[(self.buffer_index - 1) % self.BUFFER_SIZE] if self.current_sample > 0 else 0
            current_adc_a1 = self.adc_buffer_A1[(self.buffer_index - 1) % self.BUFFER_SIZE] if self.current_sample > 0 else 0
            
            # Update display with ADC diagnostics
            buffer_fill = self.buffer_index if not self.buffer_full else self.BUFFER_SIZE
            self.stats_label.setText(
                f'Samples: {self.samples_received:,} | '
                f'Rate: {sample_rate:.0f} Hz | '
                f'Current ADC A0: {current_adc_a0} A1: {current_adc_a1} | '
                f'Max ADC A0: {self.max_adc_seen_A0} A1: {self.max_adc_seen_A1} | '
                f'Buffer: {buffer_fill:,}/{self.BUFFER_SIZE:,}'
            )
            
            # Reset counters periodically
            if elapsed > 2.0:  # Reset every 2 seconds for more responsive display
                self.last_update_time = current_time
                self.samples_received = 0
    
    def toggle_auto_range(self, checked):
        """Toggle auto-ranging feature"""
        self.enable_auto_range = checked
        if not checked:
            # Reset auto-range parameters
            self.signal_min = None
            self.signal_max = None
            self.auto_range_samples = 0
        print(f"Auto-ranging {'enabled' if checked else 'disabled'}")
    
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