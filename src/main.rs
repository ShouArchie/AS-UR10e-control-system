use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use serialport::SerialPortType;

use std::io::Read;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use crossbeam_channel::{unbounded, Receiver, Sender};
use ringbuffer::{AllocRingBuffer, RingBuffer};
use realfft::RealFftPlanner;


// High-performance circular buffer for voltage data - OPTIMIZED FOR 60kHz
const BUFFER_SIZE: usize = 2_000_000; // 10 seconds at 200kHz
const DISPLAY_SAMPLES: usize = 100_000; // Show last 0.5 seconds at 200kHz
const FFT_SIZE: usize = 8192;
const UPDATE_RATE_MS: u64 = 16; // ~60 FPS; GUI overhead down

const SAMPLES_PER_BATCH: usize = 2000; // Must match firmware (0.01s batches)
const PACKET_SIZE: usize = 2 + 4 + SAMPLES_PER_BATCH * 2; // 4006 bytes

#[derive(Clone, Debug)]
struct DataBatch {
    voltages: Vec<f32>,
    batch_id: u32,
    timestamp: Instant,
}

struct HighPerfDataCollector {
    // Circular buffer for high-speed data
    voltage_buffer: Arc<Mutex<AllocRingBuffer<f32>>>,
    time_buffer: Arc<Mutex<AllocRingBuffer<f64>>>,
    
    // Statistics
    sample_rate: f64,
    total_samples: Arc<Mutex<u64>>,
    batch_count: Arc<Mutex<u32>>,
    last_update: Arc<Mutex<Instant>>,
    
    // Global time tracking
    start_time: Arc<Mutex<Option<Instant>>>,
    current_time_offset: Arc<Mutex<f64>>,
    
    // Communication channels
    data_sender: Sender<DataBatch>,
    data_receiver: Receiver<DataBatch>,
    
    // Serial connection
    running: Arc<Mutex<bool>>,
}

impl HighPerfDataCollector {
    fn new() -> Self {
        let (sender, receiver) = unbounded();
        
        Self {
            voltage_buffer: Arc::new(Mutex::new(AllocRingBuffer::new(BUFFER_SIZE))),
            time_buffer: Arc::new(Mutex::new(AllocRingBuffer::new(BUFFER_SIZE))),
            sample_rate: 200000.0, // Updated to 200kHz
            total_samples: Arc::new(Mutex::new(0)),
            batch_count: Arc::new(Mutex::new(0)),
            last_update: Arc::new(Mutex::new(Instant::now())),
            start_time: Arc::new(Mutex::new(None)),
            current_time_offset: Arc::new(Mutex::new(0.0)),
            data_sender: sender,
            data_receiver: receiver,
            running: Arc::new(Mutex::new(false)),
        }
    }
    
    fn find_pico_port() -> Option<String> {
        match serialport::available_ports() {
            Ok(ports) => {
                for port in &ports {
                    if let SerialPortType::UsbPort(usb_info) = &port.port_type {
                        if usb_info.vid == 0x2E8A { // Pico VID
                            println!("Found Pico on {}", port.port_name);
                            return Some(port.port_name.clone());
                        }
                    }
                }
                // Fallback to first port
                if !ports.is_empty() {
                    println!("Using first port: {}", ports[0].port_name);
                    return Some(ports[0].port_name.clone());
                }
            }
            Err(e) => println!("Error scanning ports: {}", e),
        }
        None
    }
    
    fn start_collection(&self, port_name: String) {
        // ----- Reset statistics for a fresh session -----
        *self.total_samples.lock().unwrap() = 0;
        *self.batch_count.lock().unwrap() = 0;
        *self.start_time.lock().unwrap() = None;
        *self.current_time_offset.lock().unwrap() = 0.0;

        *self.running.lock().unwrap() = true;
        
        // Clone necessary data for the thread
        let sender = self.data_sender.clone();
        let running = self.running.clone();
        
        thread::spawn(move || {
            println!("Starting high-speed data collection on {}", port_name);
            
            let mut port = match serialport::new(&port_name, 12_000_000) // 12M baud for maximum USB throughput
                .timeout(Duration::from_millis(100)) // More headroom for full packet arrival
                .open()
            {
                Ok(port) => port,
                Err(e) => {
                    println!("Failed to open port: {}", e);
                    return;
                }
            };
            
            let mut buf = [0u8; PACKET_SIZE];
            let mut last_batch_time = Instant::now();
            let mut synced = false;
            
            while *running.lock().unwrap() {
                // --- Synchronisation phase ---
                if !synced {
                    // Search for 0xA5 0x5A sequence
                    let mut byte = [0u8; 1];
                    // read until first 0xA5 found
                    loop {
                        if let Err(e) = port.read_exact(&mut byte) {
                            println!("Serial read error: {}", e);
                            thread::sleep(Duration::from_millis(1));
                            continue;
                        }
                        if byte[0] == 0xA5 { break; }
                    }
                    // read second byte
                    if let Err(e) = port.read_exact(&mut byte) {
                        println!("Serial read error: {}", e);
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    if byte[0] != 0x5A { continue; }

                    // Header matched, read rest of packet (PACKET_SIZE-2)
                    if let Err(e) = port.read_exact(&mut buf[2..]) {
                        println!("Serial read error: {}", e);
                        continue;
                    }
                    buf[0] = 0xA5;
                    buf[1] = 0x5A;
                    synced = true;
                } else {
                    // Already synced: header already consumed in previous loop, so read full packet
                    if let Err(e) = port.read_exact(&mut buf) {
                        println!("Serial read error: {}", e);
                        synced = false;
                        continue;
                    }
                    // Quick check header
                    if buf[0] != 0xA5 || buf[1] != 0x5A {
                        println!("Desync detected");
                        synced = false;
                        continue;
                    }
                }

                let id_bytes = &buf[2..6];
                let id = u32::from_le_bytes([id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]]);

                // Convert samples
                let mut voltages = Vec::with_capacity(SAMPLES_PER_BATCH);
                for i in 0..SAMPLES_PER_BATCH {
                    let offset = 6 + i * 2;
                    let sample = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
                    let voltage = sample as f32 * 3.3 / 4095.0;
                    voltages.push(voltage);
                }

                let batch = DataBatch {
                    voltages,
                    timestamp: last_batch_time,
                    batch_id: id,
                };

                if sender.try_send(batch).is_err() {
                    println!("Warning: Data processing falling behind");
                }

                last_batch_time = Instant::now();
            }
            
            println!("Data collection thread stopped");
        });
        
        // Start data processing thread
        self.start_processing();
    }
    
    fn start_processing(&self) {
        let receiver = self.data_receiver.clone();
        let voltage_buffer = self.voltage_buffer.clone();
        let time_buffer = self.time_buffer.clone();
        let total_samples = self.total_samples.clone();
        let batch_count = self.batch_count.clone();
        let last_update = self.last_update.clone();
        let start_time = self.start_time.clone();
        let current_time_offset = self.current_time_offset.clone();
        let running = self.running.clone();
        let sample_rate = self.sample_rate;
        
        thread::spawn(move || {
            println!("Starting data processing thread");
            
            // Track previous batch-ID to detect missing packets
            let mut last_id: Option<u32> = None;
            // Timing diagnostics: measure wall-clock time over 100 batches
            let mut t0_wall: Option<Instant> = None;
            let mut id0: u32 = 0;
            
            while *running.lock().unwrap() {
                match receiver.recv_timeout(Duration::from_millis(25)) { // Faster processing for 60kHz
                    Ok(batch) => {
                        // --- Check for skipped batch IDs ---
                        if let Some(prev) = last_id {
                            let diff = batch.batch_id.wrapping_sub(prev);
                            if diff != 1 {
                                println!("⚠️  Batch ID jump: {} -> {} (missed {})", prev, batch.batch_id, diff.saturating_sub(1));
                            }
                        }
                        // Debug: print every batch ID received
                        println!("Received batch ID {}", batch.batch_id);
                        last_id = Some(batch.batch_id);
                        
                        let batch_size = batch.voltages.len();
                        
                        // Initialize start time on first batch
                        {
                            let mut start_time_guard = start_time.lock().unwrap();
                            if start_time_guard.is_none() {
                                *start_time_guard = Some(Instant::now());
                                println!("Initialized timing reference");
                            }
                        }
                        
                        // Get current time offset and increment it
                        let mut time_offset = {
                            let mut offset = current_time_offset.lock().unwrap();
                            let current_offset = *offset;
                            *offset += batch_size as f64 / sample_rate; // Advance by batch duration
                            current_offset
                        };
                        
                        // Lock buffers and add data with proper chronological timestamps
                        {
                            let mut v_buf = voltage_buffer.lock().unwrap();
                            let mut t_buf = time_buffer.lock().unwrap();
                            
                            for (i, &voltage) in batch.voltages.iter().enumerate() {
                                let timestamp = time_offset + (i as f64 / sample_rate);
                                v_buf.push(voltage);
                                t_buf.push(timestamp);
                            }
                        }
                        
                        // Update statistics
                        *total_samples.lock().unwrap() += batch_size as u64;
                        *batch_count.lock().unwrap() += 1;
                        *last_update.lock().unwrap() = Instant::now();
                        
                        if batch.batch_id % 10 == 0 {
                            println!("Processed batch {}: {} samples, time offset: {:.3}s", 
                                   batch.batch_id, batch_size, time_offset);
                        }

                        // ----- Wall-clock rate diagnostic -----
                        if t0_wall.is_none() {
                            t0_wall = Some(Instant::now());
                            id0 = batch.batch_id;
                        }
                        if let Some(t0) = t0_wall {
                            let batches_since = batch.batch_id.wrapping_sub(id0);
                            if batches_since == 100 {
                                let elapsed = t0.elapsed().as_secs_f64();
                                let samples = batches_since as f64 * SAMPLES_PER_BATCH as f64;
                                let rate = samples / elapsed;
                                println!("=== Diagnostic: {} batches, {:.3} s -> {:.1} samples/s ===", batches_since, elapsed, rate);
                                // reset measurement
                                t0_wall = Some(Instant::now());
                                id0 = batch.batch_id;
                            }
                        }
                    }
                    Err(_) => {
                        // Timeout - check if we should continue
                        continue;
                    }
                }
            }
            
            println!("Data processing thread stopped");
        });
    }
    
    fn stop_collection(&self) {
        *self.running.lock().unwrap() = false;
    }
    
    fn get_display_data(&self) -> (Vec<f64>, Vec<f32>) {
        let v_buf = self.voltage_buffer.lock().unwrap();
        let t_buf = self.time_buffer.lock().unwrap();
        
        let len = v_buf.len().min(t_buf.len()).min(DISPLAY_SAMPLES);
        
        if len == 0 {
            return (Vec::new(), Vec::new());
        }
        
        // Get most recent data in chronological order
        // The circular buffer maintains chronological order, so we just take the last N samples
        let total_len = v_buf.len();
        let start_idx = if total_len > len { total_len - len } else { 0 };
        
        let mut voltages = Vec::with_capacity(len);
        let mut times = Vec::with_capacity(len);
        
        // Collect data from the circular buffer in chronological order
        let v_iter = v_buf.iter().skip(start_idx);
        let t_iter = t_buf.iter().skip(start_idx);
        
        for (&voltage, &time) in v_iter.zip(t_iter) {
            voltages.push(voltage);
            times.push(time);
        }
        
        (times, voltages)
    }
    
    /// Return (total_samples, batch_count, estimated_sample_rate_hz, seconds_since_last_batch)
    fn get_stats(&self) -> (u64, u32, f64, f64) {
        let total = *self.total_samples.lock().unwrap();
        let batches = *self.batch_count.lock().unwrap();

        // ----- Compute elapsed time since first batch -----
        let elapsed = {
            let start_guard = self.start_time.lock().unwrap();
            match *start_guard {
                Some(t0) => t0.elapsed().as_secs_f64(),
                None => 0.0,
            }
        };

        // Avoid division by zero on first call(s)
        let est_rate = if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        };

        let since_last = self.last_update.lock().unwrap().elapsed().as_secs_f64();
        
        (total, batches, est_rate, since_last)
    }
}

struct PiezoMonitorApp {
    collector: HighPerfDataCollector,
    port_name: String,
    connected: bool,
    monitoring: bool,
    
    // Display settings
    downsample_factor: usize,
    auto_scale: bool,
    show_fft: bool,
    
    // FFT processing
    fft_planner: RealFftPlanner<f32>,
    fft_buffer: Vec<f32>,
    fft_output: Vec<f32>,
    fft_freqs: Vec<f64>,
}

impl PiezoMonitorApp {
    fn new() -> Self {
        let fft_planner = RealFftPlanner::new();
        let fft_buffer = vec![0.0; FFT_SIZE];
        let fft_output = vec![0.0; FFT_SIZE / 2 + 1];
        
        // Pre-calculate frequency bins for 60kHz
        let fft_freqs: Vec<f64> = (0..=FFT_SIZE/2)
            .map(|i| i as f64 * 60000.0 / FFT_SIZE as f64)
            .collect();
        
        Self {
            collector: HighPerfDataCollector::new(),
            port_name: String::new(),
            connected: false,
            monitoring: false,
            downsample_factor: 1,
            auto_scale: true,
            show_fft: true,
            fft_planner,
            fft_buffer,
            fft_output,
            fft_freqs,
        }
    }
    
    fn compute_fft(&mut self, data: &[f32]) -> Vec<[f64; 2]> {
        if data.len() < FFT_SIZE {
            return Vec::new();
        }
        
        // Copy and window the data
        let start_idx = data.len().saturating_sub(FFT_SIZE);
        for (i, &sample) in data[start_idx..].iter().enumerate() {
            self.fft_buffer[i] = sample;
        }
        
        // Apply Hann window - manual implementation since apodize API changed
        for (i, sample) in self.fft_buffer.iter_mut().enumerate() {
            let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32).cos());
            *sample *= window;
        }
        
        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(FFT_SIZE);
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut self.fft_buffer, &mut spectrum).unwrap();
        
        // Convert to magnitude and create plot points
        spectrum.iter()
            .enumerate()
            .take(FFT_SIZE / 2)
            .skip(1) // Skip DC component
            .map(|(i, complex)| {
                let magnitude = (complex.re * complex.re + complex.im * complex.im).sqrt();
                [self.fft_freqs[i], magnitude as f64]
            })
            .filter(|[freq, mag]| *freq > 0.0 && *mag > 1e-6)
            .collect()
    }
}

impl eframe::App for PiezoMonitorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request repaint for smooth updates
        ctx.request_repaint_after(Duration::from_millis(UPDATE_RATE_MS));
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🚀 ULTRA HIGH-PERFORMANCE Piezo Monitor (60kHz)");
            
            // Control panel
            ui.horizontal(|ui| {
                if !self.connected {
                    if ui.button("🔌 Connect").clicked() {
                        if let Some(port) = HighPerfDataCollector::find_pico_port() {
                            self.port_name = port;
                            self.connected = true;
                        }
                    }
                } else {
                    ui.colored_label(egui::Color32::GREEN, format!("📡 Connected: {}", self.port_name));
                    
                    if !self.monitoring {
                        if ui.button("▶️ Start").clicked() {
                            self.collector.start_collection(self.port_name.clone());
                            self.monitoring = true;
                        }
                    } else {
                        if ui.button("⏹️ Stop").clicked() {
                            self.collector.stop_collection();
                            self.monitoring = false;
                        }
                    }
                }
                
                ui.separator();
                
                ui.label("Downsample:");
                ui.add(egui::DragValue::new(&mut self.downsample_factor).range(1..=50));
                
                ui.checkbox(&mut self.auto_scale, "Auto Scale");
                ui.checkbox(&mut self.show_fft, "Show FFT");
            });
            
            // Statistics
            let (total_samples, batch_count, est_rate, since_last) = self.collector.get_stats();
            ui.horizontal(|ui| {
                ui.label(format!("📊 Samples: {}", total_samples));
                ui.label(format!("📦 Batches: {}", batch_count));
                ui.label(format!("⏱️ Rate: {:.0} Hz", est_rate));
                
                let status_color = if since_last < 2.0 { egui::Color32::GREEN } else { egui::Color32::RED };
                ui.colored_label(status_color, if since_last < 2.0 { "🟢 LIVE" } else { "🔴 STALE" });
            });
            
            ui.separator();
            
            // Get display data
            let (times, voltages) = self.collector.get_display_data();
            
            if !voltages.is_empty() {
                // Smart downsampling for 60kHz - automatically downsample for performance
                let auto_downsample = if voltages.len() > 10000 { 
                    (voltages.len() / 10000).max(self.downsample_factor)
                } else { 
                    self.downsample_factor 
                };
                
                let display_times: Vec<f64>;
                let display_voltages: Vec<f32>;
                
                if auto_downsample > 1 {
                    display_times = times.iter().step_by(auto_downsample).copied().collect();
                    display_voltages = voltages.iter().step_by(auto_downsample).copied().collect();
                } else {
                    display_times = times;
                    display_voltages = voltages.clone();
                }
                
                // Time domain plot
                let time_points: Vec<[f64; 2]> = display_times.iter()
                    .zip(display_voltages.iter())
                    .map(|(&t, &v)| [t, v as f64])
                    .collect();
                
                let mut time_plot = Plot::new("time_domain")
                    .legend(egui_plot::Legend::default())
                    .height(250.0)
                    .show_axes([true, true])
                    .show_grid(true);
                
                if self.auto_scale {
                    time_plot = time_plot.auto_bounds([true, true].into());
                }
                
                time_plot.show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::new(time_points))
                            .color(egui::Color32::BLUE)
                            .name("Voltage")
                            .width(1.0)
                    );
                });
                
                // FFT plot
                if self.show_fft && voltages.len() >= FFT_SIZE {
                    let fft_points = self.compute_fft(&voltages);
                    
                    if !fft_points.is_empty() {
                        Plot::new("frequency_domain")
                            .legend(egui_plot::Legend::default())
                            .height(250.0)
                            .show_axes([true, true])
                            .show_grid(true)
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(PlotPoints::new(fft_points))
                                        .color(egui::Color32::RED)
                                        .name("FFT Magnitude")
                                        .width(1.0)
                                );
                            });
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("📡 Waiting for data from Pico...");
                });
            }
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    println!("🚀 ULTRA HIGH-PERFORMANCE PIEZO MONITOR");
    println!("Optimized for 60kHz real-time data with circular buffers");
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("High-Performance Piezo Monitor")
            .with_resizable(true),
        ..Default::default()
    };
    
    eframe::run_native(
        "Piezo Monitor",
        options,
        Box::new(|_cc| Ok(Box::new(PiezoMonitorApp::new()))),
    )
} 