#include "hypergraph-utils.h"
#include <queue>
#include <stack>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>

namespace opencog_qat {

std::unique_ptr<HypergraphUtils::Hypergraph> HypergraphUtils::createHypergraph() {
    return std::make_unique<Hypergraph>();
}

uint32_t HypergraphUtils::addNode(Hypergraph& hg, NodeType type, const std::string& name, 
                                 const std::vector<float>& features) {
    uint32_t id = getNextId(hg);
    Node node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.features = features;
    
    hg.nodes[id] = std::move(node);
    hg.incoming_index[id] = std::vector<uint32_t>();
    
    return id;
}

uint32_t HypergraphUtils::addLink(Hypergraph& hg, LinkType type, const std::vector<uint32_t>& outgoing,
                                 float strength, float confidence) {
    uint32_t id = getNextId(hg);
    Link link;
    link.id = id;
    link.type = type;
    link.outgoing = outgoing;
    link.strength = strength;
    link.confidence = confidence;
    
    hg.links[id] = std::move(link);
    hg.outgoing_index[id] = outgoing;
    
    // Update incoming index for connected nodes
    updateIndices(hg, id, outgoing);
    
    return id;
}

std::vector<std::unordered_map<uint32_t, uint32_t>> HypergraphUtils::findPatterns(
    const Hypergraph& hg, const Hypergraph& pattern) {
    
    std::vector<std::unordered_map<uint32_t, uint32_t>> matches;
    
    // Simple pattern matching implementation
    // For each node in pattern, try to find matching nodes in hg
    if (pattern.nodes.empty()) return matches;
    
    auto pattern_node = pattern.nodes.begin();
    uint32_t pattern_start = pattern_node->first;
    
    for (const auto& [hg_node_id, hg_node] : hg.nodes) {
        if (hg_node.type == pattern_node->second.type) {
            std::unordered_map<uint32_t, uint32_t> mapping;
            mapping[pattern_start] = hg_node_id;
            
            // Try to extend the mapping
            if (extendPatternMatch(hg, pattern, mapping, pattern_start, hg_node_id)) {
                matches.push_back(mapping);
            }
        }
    }
    
    return matches;
}

std::vector<uint32_t> HypergraphUtils::breadthFirstTraversal(const Hypergraph& hg, uint32_t start_node) {
    std::vector<uint32_t> result;
    std::queue<uint32_t> queue;
    std::set<uint32_t> visited;
    
    if (hg.nodes.find(start_node) == hg.nodes.end()) {
        return result;
    }
    
    queue.push(start_node);
    visited.insert(start_node);
    
    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();
        result.push_back(current);
        
        // Get neighbors through links
        auto neighbors = getNeighbors(hg, current);
        for (uint32_t neighbor : neighbors) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }
    
    return result;
}

std::vector<uint32_t> HypergraphUtils::depthFirstTraversal(const Hypergraph& hg, uint32_t start_node) {
    std::vector<uint32_t> result;
    std::stack<uint32_t> stack;
    std::set<uint32_t> visited;
    
    if (hg.nodes.find(start_node) == hg.nodes.end()) {
        return result;
    }
    
    stack.push(start_node);
    
    while (!stack.empty()) {
        uint32_t current = stack.top();
        stack.pop();
        
        if (visited.find(current) == visited.end()) {
            visited.insert(current);
            result.push_back(current);
            
            // Get neighbors and add to stack
            auto neighbors = getNeighbors(hg, current);
            for (uint32_t neighbor : neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    stack.push(neighbor);
                }
            }
        }
    }
    
    return result;
}

std::vector<uint32_t> HypergraphUtils::shortestPath(const Hypergraph& hg, uint32_t start, uint32_t end) {
    std::vector<uint32_t> path;
    std::unordered_map<uint32_t, uint32_t> parent;
    std::unordered_map<uint32_t, float> distance;
    std::queue<uint32_t> queue;
    
    // Initialize
    distance[start] = 0.0f;
    queue.push(start);
    
    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();
        
        if (current == end) {
            // Reconstruct path
            uint32_t node = end;
            while (node != start) {
                path.push_back(node);
                node = parent[node];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            break;
        }
        
        auto neighbors = getNeighbors(hg, current);
        for (uint32_t neighbor : neighbors) {
            if (distance.find(neighbor) == distance.end()) {
                distance[neighbor] = distance[current] + 1.0f;
                parent[neighbor] = current;
                queue.push(neighbor);
            }
        }
    }
    
    return path;
}

std::vector<uint32_t> HypergraphUtils::getNeighbors(const Hypergraph& hg, uint32_t node_id) {
    std::vector<uint32_t> neighbors;
    std::set<uint32_t> unique_neighbors;
    
    // Get all links that include this node
    auto incoming_it = hg.incoming_index.find(node_id);
    if (incoming_it != hg.incoming_index.end()) {
        for (uint32_t link_id : incoming_it->second) {
            auto link_it = hg.links.find(link_id);
            if (link_it != hg.links.end()) {
                for (uint32_t connected_node : link_it->second.outgoing) {
                    if (connected_node != node_id) {
                        unique_neighbors.insert(connected_node);
                    }
                }
            }
        }
    }
    
    // Convert set to vector
    neighbors.assign(unique_neighbors.begin(), unique_neighbors.end());
    return neighbors;
}

std::unordered_map<uint32_t, float> HypergraphUtils::computeBetweennessCentrality(const Hypergraph& hg) {
    std::unordered_map<uint32_t, float> centrality;
    
    // Initialize centrality scores
    for (const auto& [node_id, node] : hg.nodes) {
        centrality[node_id] = 0.0f;
    }
    
    // For each pair of nodes, compute shortest paths and update centrality
    for (const auto& [s_id, s_node] : hg.nodes) {
        for (const auto& [t_id, t_node] : hg.nodes) {
            if (s_id != t_id) {
                auto path = shortestPath(hg, s_id, t_id);
                if (path.size() > 2) {
                    // Add centrality for intermediate nodes
                    for (size_t i = 1; i < path.size() - 1; ++i) {
                        centrality[path[i]] += 1.0f;
                    }
                }
            }
        }
    }
    
    // Normalize by number of pairs
    float num_pairs = static_cast<float>(hg.nodes.size() * (hg.nodes.size() - 1));
    for (auto& [node_id, score] : centrality) {
        score /= num_pairs;
    }
    
    return centrality;
}

std::unordered_map<uint32_t, float> HypergraphUtils::computePageRank(const Hypergraph& hg, float damping) {
    std::unordered_map<uint32_t, float> pagerank;
    std::unordered_map<uint32_t, float> new_pagerank;
    
    float num_nodes = static_cast<float>(hg.nodes.size());
    float init_value = 1.0f / num_nodes;
    
    // Initialize PageRank values
    for (const auto& [node_id, node] : hg.nodes) {
        pagerank[node_id] = init_value;
    }
    
    // Iterate until convergence
    const int max_iterations = 100;
    const float tolerance = 1e-6f;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Initialize new values
        for (const auto& [node_id, node] : hg.nodes) {
            new_pagerank[node_id] = (1.0f - damping) / num_nodes;
        }
        
        // Update PageRank values
        for (const auto& [node_id, node] : hg.nodes) {
            auto neighbors = getNeighbors(hg, node_id);
            if (!neighbors.empty()) {
                float contribution = damping * pagerank[node_id] / static_cast<float>(neighbors.size());
                for (uint32_t neighbor : neighbors) {
                    new_pagerank[neighbor] += contribution;
                }
            }
        }
        
        // Check convergence
        float diff = 0.0f;
        for (const auto& [node_id, node] : hg.nodes) {
            diff += std::abs(new_pagerank[node_id] - pagerank[node_id]);
            pagerank[node_id] = new_pagerank[node_id];
        }
        
        if (diff < tolerance) break;
    }
    
    return pagerank;
}

std::vector<std::vector<uint32_t>> HypergraphUtils::detectCommunities(const Hypergraph& hg) {
    std::vector<std::vector<uint32_t>> communities;
    
    // Simple community detection using connected components
    std::set<uint32_t> unvisited;
    for (const auto& [node_id, node] : hg.nodes) {
        unvisited.insert(node_id);
    }
    
    while (!unvisited.empty()) {
        uint32_t start = *unvisited.begin();
        auto component = breadthFirstTraversal(hg, start);
        
        std::vector<uint32_t> community;
        for (uint32_t node_id : component) {
            if (unvisited.count(node_id)) {
                community.push_back(node_id);
                unvisited.erase(node_id);
            }
        }
        
        if (!community.empty()) {
            communities.push_back(community);
        }
    }
    
    return communities;
}

std::string HypergraphUtils::serializeHypergraph(const Hypergraph& hg) {
    std::stringstream ss;
    
    // Serialize nodes
    ss << "NODES:" << hg.nodes.size() << "\\n";
    for (const auto& [id, node] : hg.nodes) {
        ss << id << "," << static_cast<int>(node.type) << "," << node.name << "\\n";
    }
    
    // Serialize links
    ss << "LINKS:" << hg.links.size() << "\\n";
    for (const auto& [id, link] : hg.links) {
        ss << id << "," << static_cast<int>(link.type);
        for (uint32_t out_id : link.outgoing) {
            ss << "," << out_id;
        }
        ss << "\\n";
    }
    
    return ss.str();
}

std::unique_ptr<HypergraphUtils::Hypergraph> HypergraphUtils::deserializeHypergraph(const std::string& data) {
    auto hg = createHypergraph();
    
    std::istringstream iss(data);
    std::string line;
    
    // Parse nodes
    if (std::getline(iss, line) && line.find("NODES:") == 0) {
        while (std::getline(iss, line) && line.find("LINKS:") != 0) {
            std::istringstream line_ss(line);
            std::string token;
            
            if (std::getline(line_ss, token, ',')) {
                uint32_t id = std::stoul(token);
                if (std::getline(line_ss, token, ',')) {
                    NodeType type = static_cast<NodeType>(std::stoi(token));
                    if (std::getline(line_ss, token)) {
                        addNode(*hg, type, token);
                    }
                }
            }
        }
    }
    
    // Parse links
    while (std::getline(iss, line)) {
        std::istringstream line_ss(line);
        std::string token;
        
        if (std::getline(line_ss, token, ',')) {
            uint32_t id = std::stoul(token);
            if (std::getline(line_ss, token, ',')) {
                LinkType type = static_cast<LinkType>(std::stoi(token));
                std::vector<uint32_t> outgoing;
                
                while (std::getline(line_ss, token, ',')) {
                    outgoing.push_back(std::stoul(token));
                }
                
                addLink(*hg, type, outgoing);
            }
        }
    }
    
    return hg;
}

std::unique_ptr<HypergraphUtils::Hypergraph> HypergraphUtils::mergeHypergraphs(
    const Hypergraph& hg1, const Hypergraph& hg2) {
    
    auto merged = createHypergraph();
    std::unordered_map<uint32_t, uint32_t> id_mapping;
    
    // Copy nodes from hg1
    for (const auto& [id, node] : hg1.nodes) {
        uint32_t new_id = addNode(*merged, node.type, node.name, node.features);
        id_mapping[id] = new_id;
    }
    
    // Copy nodes from hg2 (with potential renaming)
    for (const auto& [id, node] : hg2.nodes) {
        uint32_t new_id = addNode(*merged, node.type, node.name + "_2", node.features);
        id_mapping[id] = new_id;
    }
    
    // Copy links from hg1
    for (const auto& [id, link] : hg1.links) {
        std::vector<uint32_t> new_outgoing;
        for (uint32_t old_id : link.outgoing) {
            new_outgoing.push_back(id_mapping[old_id]);
        }
        addLink(*merged, link.type, new_outgoing, link.strength, link.confidence);
    }
    
    // Copy links from hg2
    for (const auto& [id, link] : hg2.links) {
        std::vector<uint32_t> new_outgoing;
        for (uint32_t old_id : link.outgoing) {
            new_outgoing.push_back(id_mapping[old_id]);
        }
        addLink(*merged, link.type, new_outgoing, link.strength, link.confidence);
    }
    
    return merged;
}

std::unique_ptr<HypergraphUtils::Hypergraph> HypergraphUtils::extractSubgraph(
    const Hypergraph& hg, const std::vector<uint32_t>& node_ids) {
    
    auto subgraph = createHypergraph();
    std::unordered_map<uint32_t, uint32_t> id_mapping;
    std::set<uint32_t> node_set(node_ids.begin(), node_ids.end());
    
    // Copy specified nodes
    for (uint32_t node_id : node_ids) {
        auto it = hg.nodes.find(node_id);
        if (it != hg.nodes.end()) {
            uint32_t new_id = addNode(*subgraph, it->second.type, it->second.name, it->second.features);
            id_mapping[node_id] = new_id;
        }
    }
    
    // Copy links that connect only the selected nodes
    for (const auto& [link_id, link] : hg.links) {
        bool all_nodes_selected = true;
        for (uint32_t out_id : link.outgoing) {
            if (node_set.find(out_id) == node_set.end()) {
                all_nodes_selected = false;
                break;
            }
        }
        
        if (all_nodes_selected) {
            std::vector<uint32_t> new_outgoing;
            for (uint32_t old_id : link.outgoing) {
                new_outgoing.push_back(id_mapping[old_id]);
            }
            addLink(*subgraph, link.type, new_outgoing, link.strength, link.confidence);
        }
    }
    
    return subgraph;
}

bool HypergraphUtils::isValidPath(const Hypergraph& hg, const std::vector<uint32_t>& path) {
    if (path.size() < 2) return true;
    
    for (size_t i = 0; i < path.size() - 1; ++i) {
        uint32_t current = path[i];
        uint32_t next = path[i + 1];
        
        // Check if there's a connection between current and next
        auto neighbors = getNeighbors(hg, current);
        if (std::find(neighbors.begin(), neighbors.end(), next) == neighbors.end()) {
            return false;
        }
    }
    
    return true;
}

HypergraphUtils::HypergraphStats HypergraphUtils::computeStatistics(const Hypergraph& hg) {
    HypergraphStats stats;
    
    stats.num_nodes = hg.nodes.size();
    stats.num_links = hg.links.size();
    
    // Count node types
    for (const auto& [node_id, node] : hg.nodes) {
        stats.node_type_counts[node.type]++;
    }
    
    // Count link types
    for (const auto& [link_id, link] : hg.links) {
        stats.link_type_counts[link.type]++;
    }
    
    // Compute average degree
    float total_degree = 0.0f;
    for (const auto& [node_id, node] : hg.nodes) {
        auto neighbors = getNeighbors(hg, node_id);
        total_degree += static_cast<float>(neighbors.size());
    }
    stats.average_degree = (stats.num_nodes > 0) ? total_degree / static_cast<float>(stats.num_nodes) : 0.0f;
    
    // Compute diameter (maximum shortest path)
    stats.diameter = 0.0f;
    for (const auto& [s_id, s_node] : hg.nodes) {
        for (const auto& [t_id, t_node] : hg.nodes) {
            if (s_id != t_id) {
                auto path = shortestPath(hg, s_id, t_id);
                if (!path.empty()) {
                    stats.diameter = std::max(stats.diameter, static_cast<float>(path.size() - 1));
                }
            }
        }
    }
    
    // Compute clustering coefficient (simplified)
    stats.clustering_coefficient = 0.0f;
    // This would require more complex implementation for hypergraphs
    
    return stats;
}

bool HypergraphUtils::validateHypergraph(const Hypergraph& hg) {
    // Check that all link outgoing nodes exist
    for (const auto& [link_id, link] : hg.links) {
        for (uint32_t node_id : link.outgoing) {
            if (hg.nodes.find(node_id) == hg.nodes.end()) {
                return false;
            }
        }
    }
    
    // Check that indices are consistent
    for (const auto& [node_id, incoming_links] : hg.incoming_index) {
        if (hg.nodes.find(node_id) == hg.nodes.end()) {
            return false;
        }
        
        for (uint32_t link_id : incoming_links) {
            if (hg.links.find(link_id) == hg.links.end()) {
                return false;
            }
        }
    }
    
    return true;
}

bool HypergraphUtils::extendPatternMatch(const Hypergraph& hg, const Hypergraph& pattern,
                                        std::unordered_map<uint32_t, uint32_t>& mapping,
                                        uint32_t pattern_node, uint32_t hg_node) {
    // Simplified pattern matching extension
    // In a full implementation, this would recursively check all connected patterns
    return true;
}

// Private helper methods
void HypergraphUtils::updateIndices(Hypergraph& hg, uint32_t link_id, const std::vector<uint32_t>& outgoing) {
    for (uint32_t node_id : outgoing) {
        hg.incoming_index[node_id].push_back(link_id);
    }
}

float HypergraphUtils::computeDistance(const Node& n1, const Node& n2) {
    if (n1.features.size() != n2.features.size()) {
        return std::numeric_limits<float>::infinity();
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n1.features.size(); ++i) {
        float diff = n1.features[i] - n2.features[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

} // namespace opencog_qat