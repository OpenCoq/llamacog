#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>

namespace opencog_qat {

/**
 * CoqUtil Equivalent: Hypergraph Utilities
 * 
 * Provides utility functions for hypergraph manipulation, traversal,
 * and pattern matching in the OpenCoq/llamacog framework.
 */
class HypergraphUtils {
public:
    // Basic hypergraph node types
    enum class NodeType : uint8_t {
        CONCEPT = 0,
        PREDICATE = 1,
        LINK = 2,
        NUMBER = 3,
        VARIABLE = 4,
        LAMBDA = 5
    };

    // Hypergraph edge (link) types  
    enum class LinkType : uint8_t {
        INHERITANCE = 0,
        SIMILARITY = 1,
        IMPLICATION = 2,
        EVALUATION = 3,
        EXECUTION = 4,
        LIST = 5,
        SET = 6,
        LAMBDA_LINK = 7
    };

    struct Node {
        uint32_t id;
        NodeType type;
        std::string name;
        std::vector<float> features;  // For tensor representation
        std::unordered_map<std::string, float> properties;
    };

    struct Link {
        uint32_t id;
        LinkType type;
        std::vector<uint32_t> outgoing;  // Node IDs this link connects
        float strength = 1.0f;
        float confidence = 1.0f;
    };

    struct Hypergraph {
        std::unordered_map<uint32_t, Node> nodes;
        std::unordered_map<uint32_t, Link> links;
        std::unordered_map<uint32_t, std::vector<uint32_t>> incoming_index;  // node_id -> link_ids
        std::unordered_map<uint32_t, std::vector<uint32_t>> outgoing_index;  // link_id -> node_ids
        uint32_t next_id = 1;
    };

    /**
     * Create a new hypergraph
     */
    static std::unique_ptr<Hypergraph> createHypergraph();

    /**
     * Add a node to the hypergraph
     */
    static uint32_t addNode(Hypergraph& hg, NodeType type, const std::string& name, 
                           const std::vector<float>& features = {});

    /**
     * Add a link to the hypergraph
     */
    static uint32_t addLink(Hypergraph& hg, LinkType type, const std::vector<uint32_t>& outgoing,
                           float strength = 1.0f, float confidence = 1.0f);

    /**
     * Pattern matching in hypergraph
     */
    static std::vector<std::unordered_map<uint32_t, uint32_t>> findPatterns(
        const Hypergraph& hg, const Hypergraph& pattern);

    /**
     * Breadth-first traversal of hypergraph
     */
    static std::vector<uint32_t> breadthFirstTraversal(const Hypergraph& hg, uint32_t start_node);

    /**
     * Depth-first traversal of hypergraph
     */
    static std::vector<uint32_t> depthFirstTraversal(const Hypergraph& hg, uint32_t start_node);

    /**
     * Find shortest path between two nodes
     */
    static std::vector<uint32_t> shortestPath(const Hypergraph& hg, uint32_t start, uint32_t end);

    /**
     * Get all neighbors of a node (connected via links)
     */
    static std::vector<uint32_t> getNeighbors(const Hypergraph& hg, uint32_t node_id);

    /**
     * Compute hypergraph centrality measures
     */
    static std::unordered_map<uint32_t, float> computeBetweennessCentrality(const Hypergraph& hg);
    static std::unordered_map<uint32_t, float> computePageRank(const Hypergraph& hg, float damping = 0.85f);

    /**
     * Hypergraph clustering and community detection
     */
    static std::vector<std::vector<uint32_t>> detectCommunities(const Hypergraph& hg);

    /**
     * Serialize/deserialize hypergraph
     */
    static std::string serializeHypergraph(const Hypergraph& hg);
    static std::unique_ptr<Hypergraph> deserializeHypergraph(const std::string& data);

    /**
     * Validate hypergraph consistency
     */
    static bool validateHypergraph(const Hypergraph& hg);

    /**
     * Merge two hypergraphs
     */
    static std::unique_ptr<Hypergraph> mergeHypergraphs(const Hypergraph& hg1, const Hypergraph& hg2);

    /**
     * Create subgraph from node set
     */
    static std::unique_ptr<Hypergraph> extractSubgraph(const Hypergraph& hg, 
                                                       const std::vector<uint32_t>& node_ids);

    /**
     * Compute hypergraph statistics
     */
    struct HypergraphStats {
        size_t num_nodes;
        size_t num_links;
        float average_degree;
        float clustering_coefficient;
        float diameter;
        std::unordered_map<NodeType, size_t> node_type_counts;
        std::unordered_map<LinkType, size_t> link_type_counts;
    };

    static HypergraphStats computeStatistics(const Hypergraph& hg);

private:
    static uint32_t getNextId(Hypergraph& hg) { return hg.next_id++; }
    static void updateIndices(Hypergraph& hg, uint32_t link_id, const std::vector<uint32_t>& outgoing);
    static float computeDistance(const Node& n1, const Node& n2);
    static bool isValidPath(const Hypergraph& hg, const std::vector<uint32_t>& path);
    static bool extendPatternMatch(const Hypergraph& hg, const Hypergraph& pattern,
                                  std::unordered_map<uint32_t, uint32_t>& mapping,
                                  uint32_t pattern_node, uint32_t hg_node);
};

} // namespace opencog_qat