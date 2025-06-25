#pragma once

#include <vector>
#include <memory>
#include <random>
#include "llama.h"

namespace opencog_qat {

/**
 * Synthetic Calibration Data Generator for Data-Free QAT
 * 
 * Generates synthetic data for quantization calibration without requiring
 * actual training data, following the data-free QAT methodology.
 */
class SyntheticCalibrationGenerator {
public:
    struct CalibrationConfig {
        int batch_size = 32;
        int sequence_length = 512;
        int num_batches = 10;
        float temperature = 1.0f;
        uint32_t seed = 42;
        
        // Distribution parameters
        float activation_mean = 0.0f;
        float activation_std = 1.0f;
        
        // Layer-specific scaling
        bool use_layer_scaling = true;
        float embedding_scale = 0.8f;
        float attention_scale = 1.2f;
        float ffn_scale = 1.0f;
    };
    
    SyntheticCalibrationGenerator();
    explicit SyntheticCalibrationGenerator(const CalibrationConfig& config);
    
    /**
     * Generate synthetic activations for a specific layer type
     */
    std::vector<float> generateLayerActivations(
        const std::string& layer_name,
        const std::vector<int64_t>& shape
    );
    
    /**
     * Generate synthetic input sequences for model calibration
     */
    std::vector<std::vector<float>> generateInputSequences(
        int vocab_size,
        int num_sequences = -1  // -1 uses config batch_size
    );
    
    /**
     * Generate attention pattern-aware synthetic data
     */
    std::vector<float> generateAttentionAwareData(
        int num_heads,
        int head_dim,
        int sequence_length
    );
    
    /**
     * Generate distribution-matching synthetic embeddings
     */
    std::vector<float> generateEmbeddingData(
        int vocab_size,
        int embedding_dim
    );
    
private:
    CalibrationConfig config_;
    std::mt19937 rng_;
    std::normal_distribution<float> normal_dist_;
    std::uniform_real_distribution<float> uniform_dist_;
    
    float getLayerScale(const std::string& layer_name) const;
    void initializeDistributions();
};

/**
 * Progressive Layer-wise Quantization Manager
 * 
 * Implements progressive quantization strategy that quantizes layers
 * in a specific order to minimize accuracy degradation.
 */
class ProgressiveQuantizer {
public:
    enum class LayerType {
        EMBEDDING,
        ATTENTION_Q,
        ATTENTION_K, 
        ATTENTION_V,
        ATTENTION_O,
        FFN_GATE,
        FFN_UP,
        FFN_DOWN,
        LAYER_NORM,
        OUTPUT
    };
    
    struct QuantizationStrategy {
        LayerType layer_type;
        int target_bits;
        std::string quantization_method;  // "uniform", "row_wise", "group_wise"
        int group_size = 128;  // for group-wise quantization
        float sensitivity_threshold = 0.02f;  // 2% accuracy threshold
    };
    
    explicit ProgressiveQuantizer();
    
    /**
     * Get recommended quantization order based on sensitivity analysis
     */
    std::vector<QuantizationStrategy> getQuantizationOrder() const;
    
    /**
     * Apply quantization strategy to a specific layer
     */
    bool quantizeLayer(
        const std::string& layer_name,
        const QuantizationStrategy& strategy,
        std::vector<float>& weights,
        std::vector<float>& calibration_data
    );
    
    /**
     * Validate quantization impact on model performance
     */
    float validateQuantizationImpact(
        const std::vector<float>& original_output,
        const std::vector<float>& quantized_output
    ) const;
    
private:
    std::vector<QuantizationStrategy> default_strategies_;
    
    void initializeDefaultStrategies();
    float computeKLDivergence(
        const std::vector<float>& p,
        const std::vector<float>& q
    ) const;
};

} // namespace opencog_qat