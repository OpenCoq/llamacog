#include "extended-atomspace.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>

namespace opencog_qat {

ExtendedAtomSpace::ExtendedAtomSpace() : next_id_(1) {
    attention_.attention_budget = 1000.0f;
}

ExtendedAtomSpace::~ExtendedAtomSpace() = default;

std::shared_ptr<ExtendedAtomSpace::Node> ExtendedAtomSpace::addNode(
    HypergraphUtils::NodeType type, const std::string& name, const ExtendedTruthValue& tv) {
    
    uint32_t id = getNextId();
    auto node = std::make_shared<Node>(id, type, name);
    node->truth_value = tv;
    
    atoms_[id] = node;
    name_index_[name] = id;
    incoming_index_[id] = std::vector<uint32_t>();
    
    return node;
}

std::shared_ptr<ExtendedAtomSpace::Link> ExtendedAtomSpace::addLink(
    HypergraphUtils::LinkType type, const std::string& name,
    const std::vector<std::shared_ptr<Atom>>& outgoing, const ExtendedTruthValue& tv) {
    
    uint32_t id = getNextId();
    auto link = std::make_shared<Link>(id, type, name);
    link->truth_value = tv;
    link->outgoing = outgoing;
    
    atoms_[id] = link;
    name_index_[name] = id;
    incoming_index_[id] = std::vector<uint32_t>();
    
    updateIndices(link);
    
    return link;
}

std::shared_ptr<ExtendedAtomSpace::Atom> ExtendedAtomSpace::getAtom(uint32_t id) const {
    auto it = atoms_.find(id);
    return (it != atoms_.end()) ? it->second : nullptr;
}

std::shared_ptr<ExtendedAtomSpace::Atom> ExtendedAtomSpace::getAtom(const std::string& name) const {
    auto it = name_index_.find(name);
    if (it != name_index_.end()) {
        return getAtom(it->second);
    }
    return nullptr;
}

bool ExtendedAtomSpace::removeAtom(uint32_t id) {
    auto it = atoms_.find(id);
    if (it == atoms_.end()) {
        return false;
    }
    
    auto atom = it->second;
    
    // Remove from name index
    name_index_.erase(atom->name);
    
    // Remove from indices
    removeFromIndices(id);
    
    // Remove from attention
    attention_.sti_values.erase(id);
    attention_.lti_values.erase(id);
    attention_.vlti_values.erase(id);
    auto focus_it = std::find(attention_.focused_atoms.begin(), attention_.focused_atoms.end(), id);
    if (focus_it != attention_.focused_atoms.end()) {
        attention_.focused_atoms.erase(focus_it);
    }
    
    // Remove from atoms map
    atoms_.erase(it);
    
    return true;
}

std::vector<std::shared_ptr<ExtendedAtomSpace::Atom>> ExtendedAtomSpace::getAllAtoms() const {
    std::vector<std::shared_ptr<Atom>> result;
    for (const auto& [id, atom] : atoms_) {
        result.push_back(atom);
    }
    return result;
}

std::vector<std::shared_ptr<ExtendedAtomSpace::Node>> ExtendedAtomSpace::getAllNodes() const {
    std::vector<std::shared_ptr<Node>> result;
    for (const auto& [id, atom] : atoms_) {
        if (atom->isNode()) {
            result.push_back(std::static_pointer_cast<Node>(atom));
        }
    }
    return result;
}

std::vector<std::shared_ptr<ExtendedAtomSpace::Link>> ExtendedAtomSpace::getAllLinks() const {
    std::vector<std::shared_ptr<Link>> result;
    for (const auto& [id, atom] : atoms_) {
        if (atom->isLink()) {
            result.push_back(std::static_pointer_cast<Link>(atom));
        }
    }
    return result;
}

bool ExtendedAtomSpace::setAtomTensor(uint32_t atom_id, const std::vector<float>& tensor_data, 
                                     const TensorShape& shape) {
    auto atom = getAtom(atom_id);
    if (!atom) return false;
    
    if (!isValidTensorShape(shape)) return false;
    
    if (tensor_data.size() != shape.totalSize()) return false;
    
    atom->tensor_data = tensor_data;
    atom->tensor_shape = shape;
    
    return true;
}

std::vector<float> ExtendedAtomSpace::getAtomTensor(uint32_t atom_id) const {
    auto atom = getAtom(atom_id);
    return atom ? atom->tensor_data : std::vector<float>();
}

ExtendedAtomSpace::TensorShape ExtendedAtomSpace::getAtomTensorShape(uint32_t atom_id) const {
    auto atom = getAtom(atom_id);
    return atom ? atom->tensor_shape : TensorShape();
}

void ExtendedAtomSpace::initializeNeuralTensorSpecs(size_t N, size_t D, size_t F, size_t A, size_t T,
                                                   size_t L, size_t P, size_t G, size_t S, size_t E,
                                                   size_t C, size_t R, size_t M, size_t goal_categories) {
    // Phase 2: Neural Integration
    tensor_specs_.neuron_tensor.dimensions = {N, D, F};
    tensor_specs_.attention_tensor.dimensions = {A, T};
    
    // Phase 3: Advanced Reasoning
    tensor_specs_.pln_tensor.dimensions = {L, P};
    tensor_specs_.moses_tensor.dimensions = {G, S, E};
    tensor_specs_.causal_tensor.dimensions = {C, L};  // Reusing L for logical chain length
    
    // Phase 4: Emergent Capabilities
    tensor_specs_.meta_tensor.dimensions = {R, M};
    tensor_specs_.goal_tensor.dimensions = {goal_categories, C};  // Reusing C for cognitive context
}

ExtendedAtomSpace::QueryResult ExtendedAtomSpace::query(const Pattern& pattern) const {
    QueryResult result;
    
    // Simple pattern matching implementation
    for (const auto& [id, atom] : atoms_) {
        float match_score = computePatternMatch(atom, pattern);
        if (match_score > 0.5f) {  // Threshold for match
            std::unordered_map<std::string, std::shared_ptr<Atom>> binding;
            binding["match"] = atom;
            result.bindings.push_back(binding);
        }
    }
    
    result.confidence = result.bindings.empty() ? 0.0f : 1.0f;
    return result;
}

std::vector<std::shared_ptr<ExtendedAtomSpace::Atom>> ExtendedAtomSpace::getIncoming(uint32_t atom_id) const {
    std::vector<std::shared_ptr<Atom>> result;
    auto it = incoming_index_.find(atom_id);
    if (it != incoming_index_.end()) {
        for (uint32_t link_id : it->second) {
            auto atom = getAtom(link_id);
            if (atom) {
                result.push_back(atom);
            }
        }
    }
    return result;
}

std::vector<std::shared_ptr<ExtendedAtomSpace::Atom>> ExtendedAtomSpace::getOutgoing(uint32_t atom_id) const {
    std::vector<std::shared_ptr<Atom>> result;
    auto atom = getAtom(atom_id);
    if (atom && atom->isLink()) {
        auto link = std::static_pointer_cast<Link>(atom);
        result = link->outgoing;
    }
    return result;
}

bool ExtendedAtomSpace::encodeNeuralActivations(const std::vector<float>& activations, 
                                               const std::string& layer_name,
                                               const TensorShape& shape) {
    // Create a concept node for the layer
    auto layer_node = addNode(HypergraphUtils::NodeType::CONCEPT, layer_name + "_activations");
    
    // Set the tensor data
    if (!setAtomTensor(layer_node->id, activations, shape)) {
        return false;
    }
    
    // Create links to represent the neural encoding
    std::vector<std::shared_ptr<Atom>> outgoing = {layer_node};
    auto encoding_link = addLink(HypergraphUtils::LinkType::EVALUATION, 
                                layer_name + "_neural_encoding", outgoing);
    
    return true;
}

void ExtendedAtomSpace::updateAttention(const AttentionAllocation& allocation) {
    attention_ = allocation;
}

ExtendedAtomSpace::AttentionAllocation ExtendedAtomSpace::getCurrentAttention() const {
    return attention_;
}

std::vector<uint32_t> ExtendedAtomSpace::getFocusedAtoms(size_t max_count) const {
    std::vector<std::pair<uint32_t, float>> sti_pairs;
    
    for (const auto& [atom_id, sti] : attention_.sti_values) {
        sti_pairs.emplace_back(atom_id, sti);
    }
    
    // Sort by STI value (descending)
    std::sort(sti_pairs.begin(), sti_pairs.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<uint32_t> result;
    size_t count = std::min(max_count, sti_pairs.size());
    for (size_t i = 0; i < count; ++i) {
        result.push_back(sti_pairs[i].first);
    }
    
    return result;
}

float ExtendedAtomSpace::evaluateFormula(std::shared_ptr<Atom> formula) const {
    if (!formula) return 0.0f;
    
    // Simple evaluation based on truth value
    return formula->truth_value.strength * formula->truth_value.confidence;
}

std::vector<std::shared_ptr<ExtendedAtomSpace::Atom>> ExtendedAtomSpace::performInference(
    const std::vector<std::shared_ptr<Atom>>& premises) const {
    
    std::vector<std::shared_ptr<Atom>> conclusions;
    
    // Simple inference: if we have premises, generate some basic conclusions
    for (auto premise : premises) {
        if (premise && premise->isLink()) {
            auto link = std::static_pointer_cast<Link>(premise);
            if (link->link_type == HypergraphUtils::LinkType::IMPLICATION && 
                link->outgoing.size() == 2) {
                
                // Modus ponens: A -> B, A ⊢ B
                auto antecedent = link->outgoing[0];
                auto consequent = link->outgoing[1];
                
                // Check if antecedent is true
                if (antecedent->truth_value.strength > 0.5f) {
                    conclusions.push_back(consequent);
                }
            }
        }
    }
    
    return conclusions;
}

ExtendedAtomSpace::ExtendedTruthValue ExtendedAtomSpace::computeConjunction(
    const ExtendedTruthValue& tv1, const ExtendedTruthValue& tv2) const {
    
    ExtendedTruthValue result;
    result.strength = tv1.strength * tv2.strength;
    result.confidence = std::min(tv1.confidence, tv2.confidence);
    result.count = std::min(tv1.count, tv2.count);
    result.uncertainty = std::max(tv1.uncertainty, tv2.uncertainty);
    
    return result;
}

ExtendedAtomSpace::ExtendedTruthValue ExtendedAtomSpace::computeDisjunction(
    const ExtendedTruthValue& tv1, const ExtendedTruthValue& tv2) const {
    
    ExtendedTruthValue result;
    result.strength = tv1.strength + tv2.strength - (tv1.strength * tv2.strength);
    result.confidence = std::max(tv1.confidence, tv2.confidence);
    result.count = std::max(tv1.count, tv2.count);
    result.uncertainty = std::min(tv1.uncertainty, tv2.uncertainty);
    
    return result;
}

ExtendedAtomSpace::ExtendedTruthValue ExtendedAtomSpace::computeNegation(
    const ExtendedTruthValue& tv) const {
    
    ExtendedTruthValue result;
    result.strength = 1.0f - tv.strength;
    result.confidence = tv.confidence;
    result.count = tv.count;
    result.uncertainty = tv.uncertainty;
    
    return result;
}

ExtendedAtomSpace::ExtendedTruthValue ExtendedAtomSpace::computeImplication(
    const ExtendedTruthValue& antecedent, const ExtendedTruthValue& consequent) const {
    
    ExtendedTruthValue result;
    
    // P(B|A) = P(A ∧ B) / P(A)
    if (antecedent.strength > 0.001f) {
        auto conjunction = computeConjunction(antecedent, consequent);
        result.strength = conjunction.strength / antecedent.strength;
    } else {
        result.strength = 1.0f;  // Vacuous truth
    }
    
    result.confidence = std::min(antecedent.confidence, consequent.confidence);
    result.count = std::min(antecedent.count, consequent.count);
    result.uncertainty = std::max(antecedent.uncertainty, consequent.uncertainty);
    
    return result;
}

ExtendedAtomSpace::AtomSpaceStats ExtendedAtomSpace::getStatistics() const {
    AtomSpaceStats stats;
    
    stats.total_atoms = atoms_.size();
    stats.total_nodes = 0;
    stats.total_links = 0;
    stats.total_tensor_elements = 0;
    
    float sum_strength = 0.0f;
    float sum_confidence = 0.0f;
    
    for (const auto& [id, atom] : atoms_) {
        if (atom->isNode()) {
            stats.total_nodes++;
        } else {
            stats.total_links++;
        }
        
        stats.type_counts[atom->getTypeName()]++;
        sum_strength += atom->truth_value.strength;
        sum_confidence += atom->truth_value.confidence;
        stats.total_tensor_elements += atom->tensor_data.size();
    }
    
    stats.average_truth_strength = (stats.total_atoms > 0) ? sum_strength / stats.total_atoms : 0.0f;
    stats.average_confidence = (stats.total_atoms > 0) ? sum_confidence / stats.total_atoms : 0.0f;
    
    // Compute attention entropy
    float entropy = 0.0f;
    float total_sti = 0.0f;
    for (const auto& [id, sti] : attention_.sti_values) {
        total_sti += std::abs(sti);
    }
    
    if (total_sti > 0.0f) {
        for (const auto& [id, sti] : attention_.sti_values) {
            float prob = std::abs(sti) / total_sti;
            if (prob > 0.0f) {
                entropy -= prob * std::log2(prob);
            }
        }
    }
    stats.attention_entropy = entropy;
    
    return stats;
}

bool ExtendedAtomSpace::validate() const {
    // Check atom consistency
    for (const auto& [id, atom] : atoms_) {
        if (atom->id != id) return false;
        
        // Check links have valid outgoing atoms
        if (atom->isLink()) {
            auto link = std::static_pointer_cast<Link>(atom);
            for (auto outgoing_atom : link->outgoing) {
                if (!outgoing_atom || atoms_.find(outgoing_atom->id) == atoms_.end()) {
                    return false;
                }
            }
        }
        
        // Check tensor shape consistency
        if (!atom->tensor_data.empty() && !isValidTensorShape(atom->tensor_shape)) {
            return false;
        }
        
        if (atom->tensor_data.size() != atom->tensor_shape.totalSize()) {
            return false;
        }
    }
    
    // Check indices consistency
    for (const auto& [atom_id, incoming_links] : incoming_index_) {
        if (atoms_.find(atom_id) == atoms_.end()) return false;
        
        for (uint32_t link_id : incoming_links) {
            if (atoms_.find(link_id) == atoms_.end()) return false;
        }
    }
    
    return true;
}

std::vector<std::string> ExtendedAtomSpace::getValidationErrors() const {
    std::vector<std::string> errors;
    
    for (const auto& [id, atom] : atoms_) {
        if (atom->id != id) {
            errors.push_back("Atom ID mismatch for atom " + std::to_string(id));
        }
        
        if (atom->isLink()) {
            auto link = std::static_pointer_cast<Link>(atom);
            for (size_t i = 0; i < link->outgoing.size(); ++i) {
                auto outgoing_atom = link->outgoing[i];
                if (!outgoing_atom) {
                    errors.push_back("Link " + std::to_string(id) + " has null outgoing atom at index " + std::to_string(i));
                } else if (atoms_.find(outgoing_atom->id) == atoms_.end()) {
                    errors.push_back("Link " + std::to_string(id) + " references non-existent atom " + std::to_string(outgoing_atom->id));
                }
            }
        }
        
        if (!atom->tensor_data.empty()) {
            if (!isValidTensorShape(atom->tensor_shape)) {
                errors.push_back("Atom " + std::to_string(id) + " has invalid tensor shape");
            }
            
            if (atom->tensor_data.size() != atom->tensor_shape.totalSize()) {
                errors.push_back("Atom " + std::to_string(id) + " tensor data size doesn't match shape");
            }
        }
    }
    
    return errors;
}

// Private helper methods
void ExtendedAtomSpace::updateIndices(std::shared_ptr<Link> link) {
    for (auto outgoing_atom : link->outgoing) {
        if (outgoing_atom) {
            incoming_index_[outgoing_atom->id].push_back(link->id);
        }
    }
}

void ExtendedAtomSpace::removeFromIndices(uint32_t atom_id) {
    // Remove from incoming index
    incoming_index_.erase(atom_id);
    
    // Remove references to this atom from other atoms' incoming lists
    for (auto& [id, incoming_list] : incoming_index_) {
        incoming_list.erase(
            std::remove(incoming_list.begin(), incoming_list.end(), atom_id),
            incoming_list.end()
        );
    }
}

bool ExtendedAtomSpace::isValidTensorShape(const TensorShape& shape) const {
    if (shape.dimensions.empty()) return false;
    
    for (size_t dim : shape.dimensions) {
        if (dim == 0) return false;
    }
    
    return true;
}

float ExtendedAtomSpace::computePatternMatch(std::shared_ptr<Atom> atom, const Pattern& pattern) const {
    if (!atom || !pattern.template_atom) return 0.0f;
    
    // Simple type matching
    if (atom->getTypeName() != pattern.template_atom->getTypeName()) {
        return 0.0f;
    }
    
    // Check constraints
    float match_score = 1.0f;
    for (const auto& [var_name, constraint] : pattern.constraints) {
        if (!constraint(*atom)) {
            match_score *= 0.5f;  // Partial match
        }
    }
    
    return match_score;
}

} // namespace opencog_qat