#include "qat-framework.h"
#include "common.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

static void print_usage() {
    std::cout << R"(usage: llama-opencog-qat [options] input_model output_model

OpenCog-Aligned Hardware-Optimized Quantization-Aware Training (QAT) Framework

positional arguments:
  input_model        Path to input model file
  output_model       Path to output quantized model file

options:
  -h, --help         Show this help message and exit
  --hardware TARGET  Target hardware (cpu, gpu, tpu) [default: cpu]
  --memory LIMIT     Memory limit in GB [default: 8.0]
  --accuracy THRESH  Accuracy threshold (0.0-1.0) [default: 0.02]
  --bits MIN:MAX     Bit width range [default: 4:8]
  --enable-atomspace Enable AtomSpace quantization [default: true]
  --enable-moses     Enable MOSES quantization [default: true]
  --enable-ecan      Enable ECAN quantization [default: true]
  --batches NUM      Number of calibration batches [default: 10]
  --report PATH      Path to save optimization report [optional]

examples:
  # CPU-optimized quantization
  llama-opencog-qat --hardware cpu --memory 8.0 model.gguf model_quantized.gguf
  
  # GPU-optimized with custom settings
  llama-opencog-qat --hardware gpu --memory 16.0 --accuracy 0.015 --bits 6:8 model.gguf model_quantized.gguf
  
  # TPU-optimized with full OpenCog integration
  llama-opencog-qat --hardware tpu --memory 32.0 --enable-atomspace --enable-moses --enable-ecan model.gguf model_quantized.gguf
)";
}

int main(int argc, char** argv) {
    // Default configuration
    std::string input_model;
    std::string output_model;
    std::string target_hardware = "cpu";
    float memory_limit = 8.0f;
    float accuracy_threshold = 0.02f;
    int min_bits = 4;
    int max_bits = 8;
    bool enable_atomspace = true;
    bool enable_moses = true;
    bool enable_ecan = true;
    int num_batches = 10;
    std::string report_path;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--hardware") {
            if (++i >= argc) {
                std::cerr << "Error: --hardware requires a value" << std::endl;
                return 1;
            }
            target_hardware = argv[i];
        } else if (arg == "--memory") {
            if (++i >= argc) {
                std::cerr << "Error: --memory requires a value" << std::endl;
                return 1;
            }
            memory_limit = std::stof(argv[i]);
        } else if (arg == "--accuracy") {
            if (++i >= argc) {
                std::cerr << "Error: --accuracy requires a value" << std::endl;
                return 1;
            }
            accuracy_threshold = std::stof(argv[i]);
        } else if (arg == "--bits") {
            if (++i >= argc) {
                std::cerr << "Error: --bits requires a value" << std::endl;
                return 1;
            }
            std::string bits_str = argv[i];
            size_t colon_pos = bits_str.find(':');
            if (colon_pos != std::string::npos) {
                min_bits = std::stoi(bits_str.substr(0, colon_pos));
                max_bits = std::stoi(bits_str.substr(colon_pos + 1));
            } else {
                std::cerr << "Error: --bits format should be MIN:MAX" << std::endl;
                return 1;
            }
        } else if (arg == "--enable-atomspace") {
            enable_atomspace = true;
        } else if (arg == "--disable-atomspace") {
            enable_atomspace = false;
        } else if (arg == "--enable-moses") {
            enable_moses = true;
        } else if (arg == "--disable-moses") {
            enable_moses = false;
        } else if (arg == "--enable-ecan") {
            enable_ecan = true;
        } else if (arg == "--disable-ecan") {
            enable_ecan = false;
        } else if (arg == "--batches") {
            if (++i >= argc) {
                std::cerr << "Error: --batches requires a value" << std::endl;
                return 1;
            }
            num_batches = std::stoi(argv[i]);
        } else if (arg == "--report") {
            if (++i >= argc) {
                std::cerr << "Error: --report requires a value" << std::endl;
                return 1;
            }
            report_path = argv[i];
        } else if (arg[0] != '-') {
            // Positional arguments
            if (input_model.empty()) {
                input_model = arg;
            } else if (output_model.empty()) {
                output_model = arg;
            } else {
                std::cerr << "Error: Too many positional arguments" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown option " << arg << std::endl;
            return 1;
        }
    }
    
    // Validate required arguments
    if (input_model.empty() || output_model.empty()) {
        std::cerr << "Error: Both input_model and output_model are required" << std::endl;
        print_usage();
        return 1;
    }
    
    // Validate hardware target
    opencog_qat::OpenCogQATFramework::HardwareConstraints::TargetHardware hw_target;
    if (target_hardware == "cpu") {
        hw_target = opencog_qat::OpenCogQATFramework::HardwareConstraints::TargetHardware::CPU;
    } else if (target_hardware == "gpu") {
        hw_target = opencog_qat::OpenCogQATFramework::HardwareConstraints::TargetHardware::GPU;
    } else if (target_hardware == "tpu") {
        hw_target = opencog_qat::OpenCogQATFramework::HardwareConstraints::TargetHardware::TPU;
    } else {
        std::cerr << "Error: Invalid hardware target '" << target_hardware 
                  << "'. Must be one of: cpu, gpu, tpu" << std::endl;
        return 1;
    }
    
    // Validate ranges
    if (memory_limit <= 0.0f) {
        std::cerr << "Error: Memory limit must be positive" << std::endl;
        return 1;
    }
    
    if (accuracy_threshold <= 0.0f || accuracy_threshold >= 1.0f) {
        std::cerr << "Error: Accuracy threshold must be between 0.0 and 1.0" << std::endl;
        return 1;
    }
    
    if (min_bits < 1 || max_bits > 32 || min_bits > max_bits) {
        std::cerr << "Error: Invalid bit range. Must have 1 ≤ min_bits ≤ max_bits ≤ 32" << std::endl;
        return 1;
    }
    
    try {
        // Create QAT configuration
        opencog_qat::OpenCogQATFramework::QATConfig config;
        config.hardware_constraints.target = hw_target;
        config.hardware_constraints.memory_limit_mb = static_cast<size_t>(memory_limit * 1024);
        config.hardware_constraints.performance_threshold = accuracy_threshold;
        config.hardware_constraints.target_bit_width_min = min_bits;
        config.hardware_constraints.target_bit_width_max = max_bits;
        config.enable_atomspace_quantization = enable_atomspace;
        config.enable_moses_quantization = enable_moses;
        config.enable_ecan_quantization = enable_ecan;
        config.num_calibration_batches = num_batches;
        
        // Print configuration
        std::cout << "OpenCog QAT Framework Configuration:" << std::endl;
        std::cout << "  Input Model: " << input_model << std::endl;
        std::cout << "  Output Model: " << output_model << std::endl;
        std::cout << "  Target Hardware: " << target_hardware << std::endl;
        std::cout << "  Memory Limit: " << memory_limit << " GB" << std::endl;
        std::cout << "  Accuracy Threshold: " << (accuracy_threshold * 100.0f) << "%" << std::endl;
        std::cout << "  Bit Width Range: " << min_bits << "-" << max_bits << " bits" << std::endl;
        std::cout << "  AtomSpace Quantization: " << (enable_atomspace ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  MOSES Quantization: " << (enable_moses ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  ECAN Quantization: " << (enable_ecan ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  Calibration Batches: " << num_batches << std::endl;
        std::cout << std::endl;
        
        // Create QAT framework
        auto qat_framework = std::make_unique<opencog_qat::OpenCogQATFramework>(config);
        
        // Execute QAT framework
        bool success = qat_framework->executeQATFramework(input_model, output_model);
        
        if (!success) {
            std::cerr << "Error: QAT framework execution failed" << std::endl;
            return 1;
        }
        
        // Save optimization report if requested
        if (!report_path.empty()) {
            auto metrics = qat_framework->getLastValidationMetrics();
            auto report = qat_framework->generateOptimizationReport(metrics);
            
            std::ofstream report_file(report_path);
            if (report_file) {
                report_file << report;
                report_file.close();
                std::cout << "Optimization report saved to: " << report_path << std::endl;
            } else {
                std::cerr << "Warning: Failed to save optimization report to " << report_path << std::endl;
            }
        }
        
        std::cout << "\nQuantization completed successfully!" << std::endl;
        
        // Print summary
        auto metrics = qat_framework->getLastValidationMetrics();
        std::cout << "\nSummary:" << std::endl;
        std::cout << "  Accuracy Retention: " << (metrics.accuracy_retention * 100.0f) << "%" << std::endl;
        std::cout << "  Memory Reduction: " << (metrics.memory_reduction * 100.0f) << "%" << std::endl;
        std::cout << "  Inference Speedup: " << metrics.inference_speedup << "x" << std::endl;
        std::cout << "  OpenCog Integration: " << (metrics.opencog_integration_valid ? "Valid" : "Invalid") << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}