#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <conio.h>
#endif

class SerialReader {
private:
    volatile bool running = false;
    HANDLE reader_thread = NULL;
    CRITICAL_SECTION data_mutex;
    std::deque<float> voltage_buffer;
    std::string com_port;
    
#ifdef _WIN32
    HANDLE serial_handle = INVALID_HANDLE_VALUE;
#endif

public:
    volatile int batch_count = 0;
    volatile int total_samples = 0;
    volatile float current_rate = 0.0f;
    volatile float avg_voltage = 0.0f;
    volatile float min_voltage = 3.3f;
    volatile float max_voltage = 0.0f;
    
    SerialReader(const std::string& port) : com_port(port) {
        InitializeCriticalSection(&data_mutex);
    }
    
    ~SerialReader() {
        stop();
        DeleteCriticalSection(&data_mutex);
    }
    
    bool start() {
#ifdef _WIN32
        std::string full_port = "\\\\.\\" + com_port;
        serial_handle = CreateFileA(
            full_port.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            NULL,
            OPEN_EXISTING,
            0,
            NULL
        );
        
        if (serial_handle == INVALID_HANDLE_VALUE) {
            std::cout << "[ERROR] Failed to open " << com_port << std::endl;
            return false;
        }
        
        // Configure serial port
        DCB dcb = {0};
        dcb.DCBlength = sizeof(DCB);
        GetCommState(serial_handle, &dcb);
        dcb.BaudRate = CBR_115200;
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        SetCommState(serial_handle, &dcb);
        
        // Set timeouts
        COMMTIMEOUTS timeouts = {0};
        timeouts.ReadIntervalTimeout = 50;
        timeouts.ReadTotalTimeoutConstant = 100;
        timeouts.ReadTotalTimeoutMultiplier = 10;
        SetCommTimeouts(serial_handle, &timeouts);
        
        std::cout << "[INFO] Connected to " << com_port << std::endl;
#endif
        
        running = true;
        reader_thread = (HANDLE)_beginthreadex(NULL, 0, read_loop_wrapper, this, 0, NULL);
        return true;
    }
    
    void stop() {
        running = false;
        if (reader_thread != NULL) {
            WaitForSingleObject(reader_thread, INFINITE);
            CloseHandle(reader_thread);
            reader_thread = NULL;
        }
        
#ifdef _WIN32
        if (serial_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(serial_handle);
            serial_handle = INVALID_HANDLE_VALUE;
        }
#endif
    }
    
    std::vector<float> get_recent_data(int max_samples = 1000) {
        EnterCriticalSection(&data_mutex);
        
        int start_idx = std::max(0, (int)voltage_buffer.size() - max_samples);
        std::vector<float> result;
        
        for (size_t i = start_idx; i < voltage_buffer.size(); i++) {
            result.push_back(voltage_buffer[i]);
        }
        
        LeaveCriticalSection(&data_mutex);
        return result;
    }
    
private:
    static unsigned __stdcall read_loop_wrapper(void* param) {
        SerialReader* reader = static_cast<SerialReader*>(param);
        reader->read_loop();
        return 0;
    }
    void read_loop() {
        std::string line_buffer;
        std::vector<float> current_batch;
        bool in_batch = false;
        int batch_num = 0;
        auto last_time = std::chrono::high_resolution_clock::now();
        
        while (running) {
#ifdef _WIN32
            char buffer[1024];
            DWORD bytes_read = 0;
            
            if (ReadFile(serial_handle, buffer, sizeof(buffer) - 1, &bytes_read, NULL) && bytes_read > 0) {
                buffer[bytes_read] = '\0';
                line_buffer += std::string(buffer);
                
                // Process complete lines
                size_t pos = 0;
                while ((pos = line_buffer.find('\n')) != std::string::npos) {
                    std::string line = line_buffer.substr(0, pos);
                    line_buffer.erase(0, pos + 1);
                    
                    // Remove carriage return if present
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    
                    process_line(line, current_batch, in_batch, batch_num, last_time);
                }
            }
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void process_line(const std::string& line, std::vector<float>& current_batch, 
                     bool& in_batch, int& batch_num, 
                     std::chrono::high_resolution_clock::time_point& last_time) {
        
        if (line.find("BATCH_START:") == 0) {
            size_t pos1 = line.find(':');
            size_t pos2 = line.find(':', pos1 + 1);
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                batch_num = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
            }
            current_batch.clear();
            in_batch = true;
            
        } else if (line.find("BATCH_END:") == 0) {
            in_batch = false;
            
            if (!current_batch.empty()) {
                // Calculate rate
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();
                if (duration > 0) {
                    current_rate = (current_batch.size() * 1000.0f) / duration;
                }
                last_time = current_time;
                
                // Calculate statistics
                float sum = 0;
                float batch_min = 3.3f;
                float batch_max = 0.0f;
                
                for (float voltage : current_batch) {
                    sum += voltage;
                    batch_min = std::min(batch_min, voltage);
                    batch_max = std::max(batch_max, voltage);
                }
                
                avg_voltage = sum / current_batch.size();
                min_voltage = std::min(min_voltage.load(), batch_min);
                max_voltage = std::max(max_voltage.load(), batch_max);
                
                // Add to buffer
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    for (float voltage : current_batch) {
                        voltage_buffer.push_back(voltage);
                        if (voltage_buffer.size() > 30000) { // Keep last 1 second at 30kHz
                            voltage_buffer.pop_front();
                        }
                    }
                }
                
                batch_count++;
                total_samples += current_batch.size();
            }
            
        } else if (in_batch) {
            try {
                float voltage = std::stof(line);
                current_batch.push_back(voltage);
            } catch (...) {
                // Skip invalid lines
            }
            
        } else if (line.find("===") == 0 || line.find("Status:") == 0) {
            std::cout << "[PICO] " << line << std::endl;
        }
    }
};

class ConsoleDisplay {
private:
    SerialReader* serial_reader;
    std::atomic<bool> running{true};
    
public:
    ConsoleDisplay(SerialReader* reader) : serial_reader(reader) {}
    
    void run() {
        std::cout << "\n=== PICO ADC REAL-TIME MONITOR (C++ Fast) ===" << std::endl;
        std::cout << "Press 'q' + Enter to quit" << std::endl;
        std::cout << "Press 's' + Enter to show sample data" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        auto last_update = std::chrono::high_resolution_clock::now();
        
        // Start input monitoring thread
        std::thread input_thread([this]() {
            std::string input;
            while (running && std::getline(std::cin, input)) {
                if (input == "q" || input == "quit") {
                    running = false;
                    break;
                } else if (input == "s" || input == "show") {
                    show_sample_data();
                }
            }
        });
        
        while (running) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update).count();
            
            if (elapsed >= 500) { // Update every 500ms
                update_display();
                last_update = current_time;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (input_thread.joinable()) {
            input_thread.join();
        }
    }
    
private:
    void update_display() {
        // Clear screen (Windows)
#ifdef _WIN32
        system("cls");
#else
        system("clear");
#endif
        
        std::cout << "=== PICO ADC REAL-TIME MONITOR (C++ Fast) ===" << std::endl;
        std::cout << "Commands: 'q' = quit, 's' = show samples" << std::endl;
        std::cout << "========================================" << std::endl;
        
        if (serial_reader) {
            int batches = serial_reader->batch_count.load();
            int samples = serial_reader->total_samples.load();
            float rate = serial_reader->current_rate.load();
            float avg = serial_reader->avg_voltage.load();
            float min_v = serial_reader->min_voltage.load();
            float max_v = serial_reader->max_voltage.load();
            
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Batches Received:  " << batches << std::endl;
            std::cout << "Total Samples:     " << samples << std::endl;
            std::cout << "Current Rate:      " << std::setprecision(0) << rate << " Hz" << std::endl;
            std::cout << "Average Voltage:   " << std::setprecision(4) << avg << " V" << std::endl;
            std::cout << "Voltage Range:     " << min_v << " - " << max_v << " V" << std::endl;
            
            // Simple ASCII "graph" of recent data
            auto recent_data = serial_reader->get_recent_data(50);
            if (!recent_data.empty()) {
                std::cout << "\nRecent Voltage Levels (last 50 samples):" << std::endl;
                draw_ascii_graph(recent_data);
            }
            
            std::cout << "\nStatus: " << (rate > 0 ? "RECEIVING DATA" : "WAITING...") << std::endl;
        }
        
        std::cout << "\n> ";
    }
    
    void draw_ascii_graph(const std::vector<float>& data) {
        const int height = 10;
        const int width = 50;
        
        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());
        float range = max_val - min_val;
        
        if (range < 0.001f) range = 0.001f; // Avoid division by zero
        
        // Draw graph from top to bottom
        for (int row = height - 1; row >= 0; row--) {
            float threshold = min_val + (float)row / (height - 1) * range;
            
            std::cout << std::fixed << std::setprecision(2) << threshold << "V |";
            
            for (size_t i = 0; i < data.size() && i < width; i++) {
                if (data[i] >= threshold) {
                    std::cout << "*";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
        }
        
        std::cout << "     +";
        for (int i = 0; i < width; i++) std::cout << "-";
        std::cout << std::endl;
    }
    
    void show_sample_data() {
        auto recent_data = serial_reader->get_recent_data(20);
        std::cout << "\nLast 20 voltage samples:" << std::endl;
        for (size_t i = 0; i < recent_data.size(); i++) {
            std::cout << std::fixed << std::setprecision(4) << recent_data[i] << "V ";
            if ((i + 1) % 5 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "=== PICO ADC REAL-TIME MONITOR (C++ Fast) ===" << std::endl;
    std::cout << "Enter COM port (e.g., COM9): ";
    
    std::string com_port;
    std::getline(std::cin, com_port);
    
    if (com_port.empty()) {
        com_port = "COM9"; // Default
        std::cout << "Using default: " << com_port << std::endl;
    }
    
    SerialReader serial_reader(com_port);
    if (!serial_reader.start()) {
        std::cout << "[ERROR] Failed to connect to " << com_port << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return 1;
    }
    
    ConsoleDisplay display(&serial_reader);
    display.run();
    
    std::cout << "\nShutting down..." << std::endl;
    return 0;
} 