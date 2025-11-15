#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "colmap/util/logging.h"

namespace colmap {

std::unordered_map<int, int> ComputeNormalizedMinGraphCut(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights,
    const int num_parts) {
  THROW_CHECK(!edges.empty());
  THROW_CHECK_EQ(edges.size(), weights.size());
  THROW_CHECK_GT(num_parts, 0);

  std::vector<int> vertices;
  vertices.reserve(edges.size() * 2);
  for (const auto& edge : edges) {
    vertices.push_back(edge.first);
    vertices.push_back(edge.second);
  }

  std::sort(vertices.begin(), vertices.end());
  vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());

  std::unordered_map<int, int> labels;
  labels.reserve(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    labels.emplace(vertices[i], static_cast<int>(i % num_parts));
  }

  return labels;
}

}  // namespace colmap
