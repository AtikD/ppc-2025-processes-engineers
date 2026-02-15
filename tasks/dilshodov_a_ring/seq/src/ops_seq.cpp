#include "dilshodov_a_ring/seq/include/ops_seq.hpp"

#include <utility>

#include "dilshodov_a_ring/common/include/common.hpp"

namespace dilshodov_a_ring {

RingTopologySEQ::RingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType copy = in;
  GetInput() = std::move(copy);
  GetOutput() = {};
}

bool RingTopologySEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.source >= 0 && input.dest >= 0;
}

bool RingTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RingTopologySEQ::RunImpl() {
  GetOutput() = GetInput().data;
  return true;
}

bool RingTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dilshodov_a_ring
