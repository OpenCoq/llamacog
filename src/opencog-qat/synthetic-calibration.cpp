#include "synthetic-calibration.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace opencog_qat {

SyntheticCalibrationGenerator::SyntheticCalibrationGenerator()
    : SyntheticCalibrationGenerator(CalibrationConfig{}) {
}

SyntheticCalibrationGenerator::SyntheticCalibrationGenerator(const CalibrationConfig& config)
    : config_(config), rng_(config.seed), normal_dist_(config.activation_mean, config.activation_std),
      uniform_dist_(0.0f, 1.0f) {
    initializeDistributions();
}

void SyntheticCalibrationGenerator::initializeDistributions() {
    // Initialize random number generators with proper seeding
    rng_.seed(config_.seed);
}

float SyntheticCalibrationGenerator::getLayerScale(const std::string& layer_name) const {
    if (!config_.use_layer_scaling) {
        return 1.0f;
    }
    
    if (layer_name.find("embed") != std::string::npos) {
        return config_.embedding_scale;
    } else if (layer_name.find("attn") != std::string::npos) {
        return config_.attention_scale;
    } else if (layer_name.find("ffn") != std::string::npos || layer_name.find("mlp") != std::string::npos) {
        return config_.ffn_scale;
    }
    
    return 1.0f; // Default scale
}

std::vector<float> SyntheticCalibrationGenerator::generateLayerActivations(
    const std::string& layer_name,
    const std::vector<int64_t>& shape) {
    
    // Calculate total number of elements
    size_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= static_cast<size_t>(dim);
    }
    
    std::vector<float> activations(total_elements);
    float layer_scale = getLayerScale(layer_name);
    
    // Generate synthetic activations based on layer type
    for (size_t i = 0; i < total_elements; ++i) {
        float base_value = normal_dist_(rng_);
        
        // Apply layer-specific transformations
        if (layer_name.find("attn") != std::string::npos) {
            // Attention layers: apply softmax-like normalization
            base_value = std::tanh(base_value * layer_scale);
        } else if (layer_name.find("ffn") != std::string::npos) {
            // FFN layers: apply ReLU-like activation
            base_value = std::max(0.0f, base_value * layer_scale);
        } else {
            // Default: scaled normal distribution
            base_value *= layer_scale;
        }
        
        activations[i] = base_value;
    }
    
    return activations;
}

std::vector<std::vector<float>> SyntheticCalibrationGenerator::generateInputSequences(
    int vocab_size,
    int num_sequences) {
    
    if (num_sequences == -1) {
        num_sequences = config_.batch_size;
    }
    
    std::vector<std::vector<float>> sequences;
    sequences.reserve(num_sequences);
    
    std::uniform_int_distribution<int> token_dist(0, vocab_size - 1);
    
    for (int seq = 0; seq < num_sequences; ++seq) {
        std::vector<float> sequence;
        sequence.reserve(config_.sequence_length);
        
        for (int pos = 0; pos < config_.sequence_length; ++pos) {
            // Generate token ID and convert to float
            int token_id = token_dist(rng_);
            sequence.push_back(static_cast<float>(token_id));
        }
        
        sequences.push_back(std::move(sequence));
    }
    
    return sequences;
}

std::vector<float> SyntheticCalibrationGenerator::generateAttentionAwareData(
    int num_heads,
    int head_dim,
    int sequence_length) {
    
    size_t total_size = num_heads * head_dim * sequence_length;
    std::vector<float> attention_data(total_size);
    
    // Generate attention patterns that mimic realistic attention distributions
    for (int head = 0; head < num_heads; ++head) {
        for (int seq_pos = 0; seq_pos < sequence_length; ++seq_pos) {
            for (int dim = 0; dim < head_dim; ++dim) {
                size_t idx = head * head_dim * sequence_length + seq_pos * head_dim + dim;
                
                // Create position-aware attention patterns
                float position_bias = std::exp(-std::abs(seq_pos - sequence_length / 2) / 50.0f);
                float random_component = normal_dist_(rng_) * config_.attention_scale;
                
                attention_data[idx] = position_bias * random_component;
            }
        }
    }
    
    return attention_data;
}

std::vector<float> SyntheticCalibrationGenerator::generateEmbeddingData(
    int vocab_size,
    int embedding_dim) {
    
    size_t total_size = vocab_size * embedding_dim;
    std::vector<float> embedding_data(total_size);
    
    // Generate embeddings with realistic distribution characteristics
    float scale = config_.embedding_scale / std::sqrt(static_cast<float>(embedding_dim));
    
    for (size_t i = 0; i < total_size; ++i) {
        embedding_data[i] = normal_dist_(rng_) * scale;
    }
    
    return embedding_data;
}

// ProgressiveQuantizer Implementation

ProgressiveQuantizer::ProgressiveQuantizer() {
    initializeDefaultStrategies();
}

void ProgressiveQuantizer::initializeDefaultStrategies() {
    // Define quantization strategies based on the problem specification
    default_strategies_ = {
        // Embeddings: 8-bit uniform quantization
        {LayerType::EMBEDDING, 8, "uniform", 0, 0.01f},
        
        // Layer norms: 8-bit uniform quantization  
        {LayerType::LAYER_NORM, 8, "uniform", 0, 0.01f},
        
        // FFN layers: 6-bit group-wise quantization
        {LayerType::FFN_GATE, 6, "group_wise", 128, 0.015f},
        {LayerType::FFN_UP, 6, "group_wise", 128, 0.015f},
        {LayerType::FFN_DOWN, 6, "group_wise", 128, 0.015f},
        
        // Attention: 4-bit row-wise quantization (most sensitive, quantized last)
        {LayerType::ATTENTION_Q, 4, "row_wise", 0, 0.02f},
        {LayerType::ATTENTION_K, 4, "row_wise", 0, 0.02f},
        {LayerType::ATTENTION_V, 4, "row_wise", 0, 0.02f},
        {LayerType::ATTENTION_O, 4, "row_wise", 0, 0.02f},
        
        // Output layer (most sensitive)
        {LayerType::OUTPUT, 8, "uniform", 0, 0.02f}
    };
}

std::vector<ProgressiveQuantizer::QuantizationStrategy> ProgressiveQuantizer::getQuantizationOrder() const {
    return default_strategies_;
}

bool ProgressiveQuantizer::quantizeLayer(
    const std::string& layer_name,
    const QuantizationStrategy& strategy,
    std::vector<float>& weights,
    std::vector<float>& calibration_data) {
    
    if (weights.empty()) {
        std::cerr << "Error: Empty weights vector for layer " << layer_name << std::endl;
        return false;
    }
    
    // Store original weights for validation
    std::vector<float> original_weights = weights;
    
    // Apply quantization based on strategy
    if (strategy.quantization_method == "uniform") {
        // Uniform quantization
        float min_val = *std::min_element(weights.begin(), weights.end());
        float max_val = *std::max_element(weights.begin(), weights.end());
        
        int levels = (1 << strategy.target_bits) - 1;
        float scale = (max_val - min_val) / levels;
        
        for (auto& weight : weights) {
            int quantized = static_cast<int>((weight - min_val) / scale + 0.5f);
            quantized = std::clamp(quantized, 0, levels);
            weight = min_val + quantized * scale;
        }
        
    } else if (strategy.quantization_method == "row_wise") {
        // Row-wise quantization (assuming weights are in row-major format)
        // For simplicity, treat as blocks of 256 elements
        size_t block_size = std::min(static_cast<size_t>(256), weights.size());
        
        for (size_t start = 0; start < weights.size(); start += block_size) {
            size_t end = std::min(start + block_size, weights.size());
            
            auto min_it = std::min_element(weights.begin() + start, weights.begin() + end);
            auto max_it = std::max_element(weights.begin() + start, weights.begin() + end);
            
            float min_val = *min_it;
            float max_val = *max_it;
            
            int levels = (1 << strategy.target_bits) - 1;
            float scale = (max_val - min_val) / levels;
            
            for (size_t i = start; i < end; ++i) {
                int quantized = static_cast<int>((weights[i] - min_val) / scale + 0.5f);
                quantized = std::clamp(quantized, 0, levels);
                weights[i] = min_val + quantized * scale;
            }
        }
        
    } else if (strategy.quantization_method == "group_wise") {
        // Group-wise quantization
        size_t group_size = static_cast<size_t>(strategy.group_size);
        
        for (size_t start = 0; start < weights.size(); start += group_size) {
            size_t end = std::min(start + group_size, weights.size());
            
            auto min_it = std::min_element(weights.begin() + start, weights.begin() + end);
            auto max_it = std::max_element(weights.begin() + start, weights.begin() + end);
            
            float min_val = *min_it;
            float max_val = *max_it;
            
            int levels = (1 << strategy.target_bits) - 1;
            float scale = (max_val - min_val) / levels;
            
            for (size_t i = start; i < end; ++i) {
                int quantized = static_cast<int>((weights[i] - min_val) / scale + 0.5f);
                quantized = std::clamp(quantized, 0, levels);
                weights[i] = min_val + quantized * scale;
            }
        }
    }
    
    // Validate quantization impact
    float impact = validateQuantizationImpact(original_weights, weights);
    
    if (impact > strategy.sensitivity_threshold) {
        std::cerr << "Warning: Quantization impact (" << impact 
                  << ") exceeds threshold (" << strategy.sensitivity_threshold 
                  << ") for layer " << layer_name << std::endl;
        // Optionally revert quantization if impact is too high
        // weights = original_weights;
        // return false;
    }
    
    return true;
}

float ProgressiveQuantizer::validateQuantizationImpact(
    const std::vector<float>& original_output,
    const std::vector<float>& quantized_output) const {
    
    if (original_output.size() != quantized_output.size()) {
        return 1.0f; // Maximum error if sizes don't match
    }
    
    // Compute KL divergence as impact metric
    return computeKLDivergence(original_output, quantized_output);
}

float ProgressiveQuantizer::computeKLDivergence(
    const std::vector<float>& p,
    const std::vector<float>& q) const {
    
    if (p.size() != q.size() || p.empty()) {
        return 1.0f;
    }
    
    // Convert to probability distributions
    std::vector<float> p_norm(p.size()), q_norm(q.size());
    
    // Softmax normalization for p
    float p_max = *std::max_element(p.begin(), p.end());
    float p_sum = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) {
        p_norm[i] = std::exp(p[i] - p_max);
        p_sum += p_norm[i];
    }
    for (auto& val : p_norm) val /= p_sum;
    
    // Softmax normalization for q
    float q_max = *std::max_element(q.begin(), q.end());
    float q_sum = 0.0f;
    for (size_t i = 0; i < q.size(); ++i) {
        q_norm[i] = std::exp(q[i] - q_max);
        q_sum += q_norm[i];
    }
    for (auto& val : q_norm) val /= q_sum;
    
    // Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    float kl_div = 0.0f;
    const float epsilon = 1e-10f; // Avoid log(0)
    
    for (size_t i = 0; i < p.size(); ++i) {
        float p_val = std::max(p_norm[i], epsilon);
        float q_val = std::max(q_norm[i], epsilon);
        float log_ratio = std::log(p_val / q_val);
        if (std::isfinite(log_ratio)) {
            kl_div += p_val * log_ratio;
        }
    }
    
    return std::max(0.0f, kl_div); // Ensure non-negative result
}

} // namespace opencog_qat