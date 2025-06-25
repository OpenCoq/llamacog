#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

namespace opencog_qat {

/**
 * AtomSpace Quantization Module
 * 
 * Handles quantization of OpenCog AtomSpace components including
 * hypergraph structures, atom types, and truth values.
 */
class AtomSpaceQuantizer {
public:
    struct TruthValue {
        float strength;
        float confidence;
        float count;
    };
    
    struct QuantizedTruthValue {
        uint8_t strength;    // 8-bit uniform quantization
        uint8_t confidence;  
        uint8_t count;
        
        // Quantization parameters
        float strength_scale;
        float strength_offset;
        float confidence_scale;
        float confidence_offset;
        float count_scale;
        float count_offset;
    };
    
    struct AtomNode {
        int32_t atom_id;
        uint16_t atom_type;
        TruthValue truth_value;
        std::vector<int32_t> incoming_links;
        std::vector<int32_t> outgoing_links;
    };
    
    explicit AtomSpaceQuantizer();
    
    /**
     * Quantize truth values using 8-bit uniform quantization
     */
    QuantizedTruthValue quantizeTruthValue(const TruthValue& tv);
    
    /**
     * Dequantize truth values back to float representation
     */
    TruthValue dequantizeTruthValue(const QuantizedTruthValue& qtv);
    
    /**
     * Quantize hypergraph structure while preserving indexing efficiency
     */
    std::vector<uint8_t> quantizeHypergraphStructure(
        const std::vector<AtomNode>& atoms
    );
    
    /**
     * Optimize AtomSpace indexing for quantized representation
     */
    void optimizeQuantizedIndexing(
        const std::vector<QuantizedTruthValue>& quantized_tvs
    );
    
    /**
     * Validate hypergraph traversal efficiency after quantization
     */
    bool validateTraversalEfficiency(
        const std::vector<AtomNode>& original_atoms,
        const std::vector<uint8_t>& quantized_structure
    );
    
private:
    struct QuantizationParams {
        float strength_min = 0.0f;
        float strength_max = 1.0f;
        float confidence_min = 0.0f;
        float confidence_max = 1.0f;
        float count_min = 0.0f;
        float count_max = 1000.0f;
    } params_;
    
    void calibrateQuantizationParams(const std::vector<TruthValue>& truth_values);
};

/**
 * MOSES Evolution Quantization Module
 * 
 * Handles quantization of MOSES program trees while preserving
 * genetic operation compatibility and fitness evaluation accuracy.
 */
class MOSESQuantizer {
public:
    enum class NodeType : uint8_t {
        TERMINAL = 0,
        FUNCTION = 1,
        VARIABLE = 2,
        CONSTANT = 3
    };
    
    struct ProgramNode {
        NodeType type;
        float value;           // For constants and fitness scores
        uint16_t function_id;  // For function nodes
        uint16_t variable_id;  // For variable nodes
        std::vector<int32_t> children;
    };
    
    struct QuantizedProgramNode {
        NodeType type;
        uint16_t quantized_value;  // 6-bit quantization for node values
        uint16_t function_id;
        uint16_t variable_id;
        std::vector<int32_t> children;
        
        // Quantization parameters
        float value_scale;
        float value_offset;
    };
    
    explicit MOSESQuantizer();
    
    /**
     * Quantize program tree nodes using 6-bit quantization
     */
    std::vector<QuantizedProgramNode> quantizeProgramTree(
        const std::vector<ProgramNode>& program_tree
    );
    
    /**
     * Preserve genetic operation compatibility after quantization
     */
    bool validateGeneticCompatibility(
        const std::vector<QuantizedProgramNode>& quantized_tree
    );
    
    /**
     * Optimize fitness evaluation for quantized representations
     */
    float computeQuantizedFitness(
        const std::vector<QuantizedProgramNode>& quantized_tree,
        const std::vector<float>& test_inputs
    );
    
    /**
     * Ensure evolutionary optimization stability
     */
    bool validateEvolutionaryStability(
        const std::vector<ProgramNode>& original_population,
        const std::vector<std::vector<QuantizedProgramNode>>& quantized_population
    );
    
private:
    struct QuantizationParams {
        float value_min = -10.0f;
        float value_max = 10.0f;
        int quantization_levels = 64;  // 6-bit = 2^6 = 64 levels
    } params_;
    
    void calibrateQuantizationParams(const std::vector<ProgramNode>& nodes);
    float dequantizeValue(uint16_t quantized_value, float scale, float offset);
};

/**
 * ECAN Attention Quantization Module
 * 
 * Handles quantization of ECAN (Economic Attention Network) components
 * including importance scores, STI/LTI values, and attention allocation.
 */
class ECANQuantizer {
public:
    struct AttentionValue {
        float sti;         // Short-term importance
        float lti;         // Long-term importance
        float vlti;        // Very long-term importance
        float confidence;  // Confidence in the attention value
    };
    
    struct QuantizedAttentionValue {
        uint8_t sti;       // 8-bit quantization for importance scores
        uint8_t lti;
        uint8_t vlti;
        uint8_t confidence;
        
        // Quantization parameters
        float sti_scale;
        float sti_offset;
        float lti_scale;
        float lti_offset;
        float vlti_scale;
        float vlti_offset;
        float confidence_scale;
        float confidence_offset;
    };
    
    struct AttentionAllocation {
        std::vector<int32_t> focused_atoms;
        std::vector<float> allocation_weights;
        float total_attention_budget;
    };
    
    explicit ECANQuantizer();
    
    /**
     * Quantize importance scores using 8-bit uniform quantization
     */
    QuantizedAttentionValue quantizeAttentionValue(const AttentionValue& av);
    
    /**
     * Dequantize attention values back to float representation
     */
    AttentionValue dequantizeAttentionValue(const QuantizedAttentionValue& qav);
    
    /**
     * Preserve attention allocation dynamics after quantization
     */
    bool validateAttentionDynamics(
        const std::vector<AttentionValue>& original_values,
        const std::vector<QuantizedAttentionValue>& quantized_values
    );
    
    /**
     * Maintain spreading activation precision in quantized form
     */
    std::vector<float> computeQuantizedSpreadingActivation(
        const std::vector<QuantizedAttentionValue>& quantized_values,
        const std::vector<std::vector<float>>& connectivity_matrix
    );
    
    /**
     * Optimize dynamic attention allocation for quantized representation
     */
    AttentionAllocation optimizeQuantizedAllocation(
        const std::vector<QuantizedAttentionValue>& quantized_values,
        float attention_budget
    );
    
private:
    struct QuantizationParams {
        float sti_min = -1000.0f;
        float sti_max = 1000.0f;
        float lti_min = 0.0f;
        float lti_max = 1000.0f;
        float vlti_min = 0.0f;
        float vlti_max = 100.0f;
        float confidence_min = 0.0f;
        float confidence_max = 1.0f;
    } params_;
    
    void calibrateQuantizationParams(const std::vector<AttentionValue>& attention_values);
    float computeAttentionFlow(
        const QuantizedAttentionValue& source,
        const QuantizedAttentionValue& target,
        float connection_strength
    );
};

/**
 * Integrated OpenCog Quantization Manager
 * 
 * Coordinates quantization across all OpenCog components while
 * maintaining system-level coherence and performance.
 */
class OpenCogQuantizationManager {
public:
    struct SystemMetrics {
        float pattern_mining_accuracy;
        float inference_speed;
        float memory_utilization;
        float hardware_efficiency;
        float atomspace_operation_latency;
        float moses_optimization_performance;
        float ecan_attention_dynamics_quality;
    };
    
    explicit OpenCogQuantizationManager();
    
    /**
     * Perform integrated quantization across all OpenCog components
     */
    bool quantizeOpenCogSystem(
        const std::vector<AtomSpaceQuantizer::AtomNode>& atoms,
        const std::vector<MOSESQuantizer::ProgramNode>& moses_programs,
        const std::vector<ECANQuantizer::AttentionValue>& ecan_values
    );
    
    /**
     * Validate system-level performance after quantization
     */
    SystemMetrics validateSystemPerformance();
    
    /**
     * Ensure seamless integration with existing OpenCog components
     */
    bool validateOpenCogIntegration();
    
private:
    std::unique_ptr<AtomSpaceQuantizer> atomspace_quantizer_;
    std::unique_ptr<MOSESQuantizer> moses_quantizer_;
    std::unique_ptr<ECANQuantizer> ecan_quantizer_;
    
    SystemMetrics baseline_metrics_;
    SystemMetrics current_metrics_;
    
    bool validatePerformanceThreshold(const SystemMetrics& metrics, float threshold = 0.02f);
};

} // namespace opencog_qat