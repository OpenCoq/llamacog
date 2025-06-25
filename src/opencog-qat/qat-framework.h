#pragma once

#include "synthetic-calibration.h"
#include "opencog-quantization.h"
#include <string>
#include <memory>

namespace opencog_qat {

/**
 * Hardware-Optimized, Data-Free Quantization-Aware Training Framework
 * 
 * Integrates data-free QAT with OpenCog-specific quantization to create
 * hardware-optimized models while maintaining cognitive architecture compatibility.
 */
class OpenCogQATFramework {
public:
    struct HardwareConstraints {
        enum class TargetHardware {
            CPU,
            GPU, 
            TPU
        };
        
        TargetHardware target = TargetHardware::CPU;
        size_t memory_limit_mb = 8192;  // 8GB default
        float performance_threshold = 0.02f; // 2% accuracy threshold
        float memory_reduction_target = 0.75f; // 75% reduction goal
        int target_bit_width_min = 4;
        int target_bit_width_max = 8;
    };
    
    struct QATConfig {
        // Phase 1: Data-Free QAT
        SyntheticCalibrationGenerator::CalibrationConfig calibration_config;
        
        // Phase 2: OpenCog Integration
        bool enable_atomspace_quantization = true;
        bool enable_moses_quantization = true;
        bool enable_ecan_quantization = true;
        
        // Training Protocol
        int num_calibration_batches = 10;
        float kl_divergence_threshold = 0.01f;
        bool progressive_quantization = true;
        
        // Hardware Optimization
        HardwareConstraints hardware_constraints;
    };
    
    struct ValidationMetrics {
        // Performance Metrics
        float accuracy_retention = 0.0f;
        float perplexity_ratio = 0.0f;
        float kl_divergence_loss = 0.0f;
        
        // Hardware Metrics
        float memory_reduction = 0.0f;
        float inference_speedup = 0.0f;
        float hardware_utilization = 0.0f;
        
        // OpenCog Metrics  
        OpenCogQuantizationManager::SystemMetrics opencog_metrics;
        
        // Overall Success
        bool meets_accuracy_threshold = false;
        bool meets_memory_target = false;
        bool opencog_integration_valid = false;
    };
    
    OpenCogQATFramework();
    explicit OpenCogQATFramework(const QATConfig& config);
    
    /**
     * Execute complete QAT framework pipeline
     */
    bool executeQATFramework(
        const std::string& input_model_path,
        const std::string& output_model_path
    );
    
    /**
     * Phase 1: Data-Free QAT Implementation
     */
    bool executePhase1DataFreeQAT(
        const std::string& model_path,
        std::vector<float>& model_weights
    );
    
    /**
     * Phase 2: OpenCog Integration
     */
    bool executePhase2OpenCogIntegration(
        const std::vector<float>& base_weights
    );
    
    /**
     * Generate synthetic calibration data for model
     */
    std::vector<std::vector<float>> generateCalibrationData(
        int vocab_size,
        const std::vector<std::string>& layer_names
    );
    
    /**
     * Apply progressive layer-wise quantization
     */
    bool applyProgressiveQuantization(
        std::vector<float>& model_weights,
        const std::vector<std::vector<float>>& calibration_data
    );
    
    /**
     * Validate final quantized model performance
     */
    ValidationMetrics validateQuantizedModel(
        const std::vector<float>& original_weights,
        const std::vector<float>& quantized_weights
    );
    
    /**
     * Generate hardware optimization report
     */
    std::string generateOptimizationReport(const ValidationMetrics& metrics);
    
    /**
     * Export quantized model with metadata
     */
    bool exportQuantizedModel(
        const std::string& output_path,
        const std::vector<float>& quantized_weights,
        const ValidationMetrics& metrics
    );
    
    // Getters
    const QATConfig& getConfig() const { return config_; }
    const ValidationMetrics& getLastValidationMetrics() const { return last_validation_; }
    
private:
    QATConfig config_;
    ValidationMetrics last_validation_;
    
    // Core components
    std::unique_ptr<SyntheticCalibrationGenerator> calibration_generator_;
    std::unique_ptr<ProgressiveQuantizer> progressive_quantizer_;
    std::unique_ptr<OpenCogQuantizationManager> opencog_manager_;
    
    // Helper methods
    bool initializeComponents();
    bool validateHardwareConstraints(const ValidationMetrics& metrics);
    std::vector<float> loadModelWeights(const std::string& model_path);
    bool saveModelWeights(const std::string& model_path, const std::vector<float>& weights);
    
    // Hardware-specific optimizations
    void applyHardwareOptimizations(
        std::vector<float>& weights,
        HardwareConstraints::TargetHardware target
    );
    
    // Quantization validation
    float computeAccuracyRetention(
        const std::vector<float>& original_weights,
        const std::vector<float>& quantized_weights
    );
    
    float computeMemoryReduction(
        const std::vector<float>& original_weights,
        const std::vector<float>& quantized_weights
    );
};

/**
 * Factory class for creating QAT framework instances with different configurations
 */
class QATFrameworkFactory {
public:
    /**
     * Create framework optimized for CPU deployment
     */
    static std::unique_ptr<OpenCogQATFramework> createCPUOptimized(
        float memory_limit_gb = 8.0f,
        float accuracy_threshold = 0.02f
    );
    
    /**
     * Create framework optimized for GPU deployment
     */
    static std::unique_ptr<OpenCogQATFramework> createGPUOptimized(
        float memory_limit_gb = 16.0f,
        float accuracy_threshold = 0.02f
    );
    
    /**
     * Create framework optimized for TPU deployment
     */
    static std::unique_ptr<OpenCogQATFramework> createTPUOptimized(
        float memory_limit_gb = 32.0f,
        float accuracy_threshold = 0.02f
    );
    
    /**
     * Create framework with custom configuration
     */
    static std::unique_ptr<OpenCogQATFramework> createCustom(
        const OpenCogQATFramework::QATConfig& config
    );
};

} // namespace opencog_qat