#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#endif

class SimpleSerialReader {
private:
    std::string com_port;
    HANDLE serial_handle = INVALID_HANDLE_VALUE;
    
public:
    int batch_count = 0;
    int total_samples = 0;
    float avg_voltage = 0.0f;
    float min_voltage = 3.3f;
    float max_voltage = 0.0f;
    
    SimpleSerialReader(const std::string& port) : com_port(port) {}
    
    ~SimpleSerialReader() {
        disconnect();
    }
    
    bool connect() {
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
        return true;
    }
    
    void disconnect() {
#ifdef _WIN32
        if (serial_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(serial_handle);
            serial_handle = INVALID_HANDLE_VALUE;
        }
#endif
    }
    
    std::string read_line() {
        std::string line;
        char ch;
        DWORD bytes_read;
        
#ifdef _WIN32
        while (ReadFile(serial_handle, &ch, 1, &bytes_read, NULL) && bytes_read > 0) {
            if (ch == '\n') {
                break;
            } else if (ch != '\r') {
                line += ch;
            }
        }
#endif
        return line;
    }
    
    void process_batch() {
        std::vector<float> current_batch;
        std::string line;
        bool in_batch = false;
        int batch_num = 0;
        
        std::cout << "Reading data from " << com_port << "..." << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        while (true) {
            line = read_line();
            
            if (line.empty()) {
                Sleep(10);
                continue;
            }
            
            if (line.find("BATCH_START:") == 0) {
                size_t pos1 = line.find(':');
                size_t pos2 = line.find(':', pos1 + 1);
                if (pos1 != std::string::npos && pos2 != std::string::npos) {
                    batch_num = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
                }
                current_batch.clear();
                in_batch = true;
                std::cout << "Starting batch " << batch_num << "..." << std::endl;
                
            } else if (line.find("BATCH_END:") == 0) {
                in_batch = false;
                
                if (!current_batch.empty()) {
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
                    min_voltage = std::min(min_voltage, batch_min);
                    max_voltage = std::max(max_voltage, batch_max);
                    
                    batch_count++;
                    total_samples += current_batch.size();
                    
                    // Display results
                    std::cout << std::fixed << std::setprecision(4);
                    std::cout << "Batch " << batch_num << " complete: ";
                    std::cout << current_batch.size() << " samples, ";
                    std::cout << "Avg: " << avg_voltage << "V, ";
                    std::cout << "Range: " << batch_min << "-" << batch_max << "V" << std::endl;
                    
                    // Show some sample values
                    std::cout << "Sample values: ";
                    for (size_t i = 0; i < std::min((size_t)10, current_batch.size()); i++) {
                        std::cout << current_batch[i] << "V ";
                    }
                    if (current_batch.size() > 10) {
                        std::cout << "...";
                    }
                    std::cout << std::endl;
                    std::cout << "Total batches: " << batch_count << ", Total samples: " << total_samples << std::endl;
                    std::cout << "Overall range: " << min_voltage << "-" << max_voltage << "V" << std::endl;
                    std::cout << "----------------------------------------" << std::endl;
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
    }
};

int main() {
    std::cout << "=== SIMPLE PICO ADC MONITOR ===" << std::endl;
    
    // Use COM9 directly (change this if your Pico is on a different port)
    std::string com_port = "COM9";
    std::cout << "Using COM port: " << com_port << std::endl;
    
    SimpleSerialReader reader(com_port);
    if (!reader.connect()) {
        std::cout << "[ERROR] Failed to connect to " << com_port << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return 1;
    }
    
    reader.process_batch();
    
    std::cout << "\nShutting down..." << std::endl;
    return 0;
} 