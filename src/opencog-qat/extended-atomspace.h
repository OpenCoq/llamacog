#pragma once

#include "hypergraph-utils.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>

namespace opencog_qat {

/**
 * Extended AtomSpace with full node, link, and tensor support
 * 
 * Implements a comprehensive AtomSpace that supports:
 * - Nodes and Links with truth values
 * - Tensor representations for neural-symbolic integration
 * - Pattern matching and inference
 * - Attention allocation (ECAN integration)
 */
class ExtendedAtomSpace {
public:
    // Tensor shape specifications for different components
    struct TensorShape {
        std::vector<size_t> dimensions;
        std::string format; // "NCHW", "NHWC", etc.
        
        size_t totalSize() const {
            size_t total = 1;
            for (size_t dim : dimensions) {
                total *= dim;
            }
            return total;
        }
    };

    // Neural tensor specifications as mentioned in the issue
    struct NeuralTensorSpecs {
        // Phase 2: Neural Integration tensor shapes
        TensorShape neuron_tensor{.dimensions = {0, 0, 0}, .format = "N_D_F"};  // (N, D, F) neurons/degrees/features
        TensorShape attention_tensor{.dimensions = {0, 0}, .format = "A_T"};    // (A, T) attention heads/temporal
        
        // Phase 3: Advanced Reasoning tensor shapes  
        TensorShape pln_tensor{.dimensions = {0, 0}, .format = "L_P"};          // (L, P) logic types/probability
        TensorShape moses_tensor{.dimensions = {0, 0, 0}, .format = "G_S_E"};   // (G, S, E) genome/semantic/evolutionary
        TensorShape causal_tensor{.dimensions = {0, 0}, .format = "C_L"};       // (C, L) cause-effect/logical chains
        
        // Phase 4: Emergent Capabilities tensor shapes
        TensorShape meta_tensor{.dimensions = {0, 0}, .format = "R_M"};         // (R, M) recursion depth/modifiable modules
        TensorShape goal_tensor{.dimensions = {0, 0}, .format = "G_C"};         // (G, C) goal categories/cognitive context
    };

    // Extended truth value with uncertainty
    struct ExtendedTruthValue {
        float strength = 0.0f;      // Probability/confidence in truth
        float confidence = 0.0f;    // Confidence in the strength value
        float count = 0.0f;         // Amount of evidence
        float uncertainty = 0.0f;   // Epistemic uncertainty
        
        ExtendedTruthValue() = default;
        ExtendedTruthValue(float s, float c, float cnt = 1.0f, float u = 0.0f)
            : strength(s), confidence(c), count(cnt), uncertainty(u) {}
    };

    // Atom base class
    class Atom {
    public:
        uint32_t id;
        std::string name;
        ExtendedTruthValue truth_value;
        std::vector<float> tensor_data;  // Tensor representation
        TensorShape tensor_shape;
        std::unordered_map<std::string, float> properties;
        
        virtual ~Atom() = default;
        virtual std::string getTypeName() const = 0;
        virtual bool isNode() const = 0;
        virtual bool isLink() const = 0;
    };

    // Node implementation
    class Node : public Atom {
    public:
        HypergraphUtils::NodeType node_type;
        
        Node(uint32_t id, HypergraphUtils::NodeType type, const std::string& name)
            : node_type(type) {
            this->id = id;
            this->name = name;
        }
        
        std::string getTypeName() const override {
            switch (node_type) {
                case HypergraphUtils::NodeType::CONCEPT: return "ConceptNode";
                case HypergraphUtils::NodeType::PREDICATE: return "PredicateNode";
                case HypergraphUtils::NodeType::NUMBER: return "NumberNode";
                case HypergraphUtils::NodeType::VARIABLE: return "VariableNode";
                case HypergraphUtils::NodeType::LAMBDA: return "LambdaNode";
                default: return "UnknownNode";
            }
        }
        
        bool isNode() const override { return true; }
        bool isLink() const override { return false; }
    };

    // Link implementation
    class Link : public Atom {
    public:
        HypergraphUtils::LinkType link_type;
        std::vector<std::shared_ptr<Atom>> outgoing;
        
        Link(uint32_t id, HypergraphUtils::LinkType type, const std::string& name)
            : link_type(type) {
            this->id = id;
            this->name = name;
        }
        
        std::string getTypeName() const override {
            switch (link_type) {
                case HypergraphUtils::LinkType::INHERITANCE: return "InheritanceLink";
                case HypergraphUtils::LinkType::SIMILARITY: return "SimilarityLink";
                case HypergraphUtils::LinkType::IMPLICATION: return "ImplicationLink";
                case HypergraphUtils::LinkType::EVALUATION: return "EvaluationLink";
                case HypergraphUtils::LinkType::EXECUTION: return "ExecutionLink";
                case HypergraphUtils::LinkType::LIST: return "ListLink";
                case HypergraphUtils::LinkType::SET: return "SetLink";
                case HypergraphUtils::LinkType::LAMBDA_LINK: return "LambdaLink";
                default: return "UnknownLink";
            }
        }
        
        bool isNode() const override { return false; }
        bool isLink() const override { return true; }
    };

    // Pattern for pattern matching
    struct Pattern {
        std::shared_ptr<Atom> template_atom;
        std::unordered_map<std::string, std::function<bool(const Atom&)>> constraints;
    };

    // Query results
    struct QueryResult {
        std::vector<std::unordered_map<std::string, std::shared_ptr<Atom>>> bindings;
        float confidence = 0.0f;
    };

    // Attention allocation for ECAN integration
    struct AttentionAllocation {
        std::unordered_map<uint32_t, float> sti_values;  // Short-term importance
        std::unordered_map<uint32_t, float> lti_values;  // Long-term importance
        std::unordered_map<uint32_t, float> vlti_values; // Very long-term importance
        float attention_budget = 1000.0f;
        std::vector<uint32_t> focused_atoms;
    };

    explicit ExtendedAtomSpace();
    ~ExtendedAtomSpace();

    /**
     * Core AtomSpace operations
     */
    std::shared_ptr<Node> addNode(HypergraphUtils::NodeType type, const std::string& name,
                                 const ExtendedTruthValue& tv = ExtendedTruthValue(1.0f, 1.0f));
    
    std::shared_ptr<Link> addLink(HypergraphUtils::LinkType type, const std::string& name,
                                 const std::vector<std::shared_ptr<Atom>>& outgoing,
                                 const ExtendedTruthValue& tv = ExtendedTruthValue(1.0f, 1.0f));

    std::shared_ptr<Atom> getAtom(uint32_t id) const;
    std::shared_ptr<Atom> getAtom(const std::string& name) const;
    bool removeAtom(uint32_t id);
    
    std::vector<std::shared_ptr<Atom>> getAllAtoms() const;
    std::vector<std::shared_ptr<Node>> getAllNodes() const;
    std::vector<std::shared_ptr<Link>> getAllLinks() const;

    /**
     * Tensor operations for neural-symbolic integration
     */
    bool setAtomTensor(uint32_t atom_id, const std::vector<float>& tensor_data, const TensorShape& shape);
    std::vector<float> getAtomTensor(uint32_t atom_id) const;
    TensorShape getAtomTensorShape(uint32_t atom_id) const;
    
    // Initialize tensor shapes according to issue specifications
    void initializeNeuralTensorSpecs(size_t N, size_t D, size_t F, size_t A, size_t T,
                                    size_t L, size_t P, size_t G, size_t S, size_t E,
                                    size_t C, size_t R, size_t M, size_t goal_categories);

    /**
     * Pattern matching and queries
     */
    QueryResult query(const Pattern& pattern) const;
    std::vector<std::shared_ptr<Atom>> getIncoming(uint32_t atom_id) const;
    std::vector<std::shared_ptr<Atom>> getOutgoing(uint32_t atom_id) const;
    
    /**
     * Hypergraph encoding of neural activations to AtomSpace
     */
    bool encodeNeuralActivations(const std::vector<float>& activations, 
                                const std::string& layer_name,
                                const TensorShape& shape);

    /**
     * ECAN attention allocation
     */
    void updateAttention(const AttentionAllocation& allocation);
    AttentionAllocation getCurrentAttention() const;
    std::vector<uint32_t> getFocusedAtoms(size_t max_count = 20) const;

    /**
     * Inference and reasoning
     */
    float evaluateFormula(std::shared_ptr<Atom> formula) const;
    std::vector<std::shared_ptr<Atom>> performInference(const std::vector<std::shared_ptr<Atom>>& premises) const;
    
    /**
     * Truth value operations
     */
    ExtendedTruthValue computeConjunction(const ExtendedTruthValue& tv1, const ExtendedTruthValue& tv2) const;
    ExtendedTruthValue computeDisjunction(const ExtendedTruthValue& tv1, const ExtendedTruthValue& tv2) const;
    ExtendedTruthValue computeNegation(const ExtendedTruthValue& tv) const;
    ExtendedTruthValue computeImplication(const ExtendedTruthValue& antecedent, const ExtendedTruthValue& consequent) const;

    /**
     * Statistics and metrics
     */
    struct AtomSpaceStats {
        size_t total_atoms;
        size_t total_nodes;
        size_t total_links;
        std::unordered_map<std::string, size_t> type_counts;
        float average_truth_strength;
        float average_confidence;
        size_t total_tensor_elements;
        float attention_entropy;
    };
    
    AtomSpaceStats getStatistics() const;

    /**
     * Serialization and persistence
     */
    std::string serialize() const;
    bool deserialize(const std::string& data);
    bool saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);

    /**
     * Validation and consistency checks
     */
    bool validate() const;
    std::vector<std::string> getValidationErrors() const;

    // Getter for tensor specs
    const NeuralTensorSpecs& getTensorSpecs() const { return tensor_specs_; }

private:
    std::unordered_map<uint32_t, std::shared_ptr<Atom>> atoms_;
    std::unordered_map<std::string, uint32_t> name_index_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> incoming_index_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> outgoing_index_;
    AttentionAllocation attention_;
    NeuralTensorSpecs tensor_specs_;
    uint32_t next_id_;

    // Helper methods
    uint32_t getNextId() { return next_id_++; }
    void updateIndices(std::shared_ptr<Link> link);
    void removeFromIndices(uint32_t atom_id);
    bool isValidTensorShape(const TensorShape& shape) const;
    float computePatternMatch(std::shared_ptr<Atom> atom, const Pattern& pattern) const;
};

} // namespace opencog_qat