#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dilshodov_a_ring {

struct RingMessage {
  int source = 0;
  int dest = 0;
  std::vector<int> data;
};

using InType = RingMessage;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dilshodov_a_ring
