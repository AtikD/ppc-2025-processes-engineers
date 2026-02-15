#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <random>

#include "dilshodov_a_ring/common/include/common.hpp"
#include "dilshodov_a_ring/mpi/include/ops_mpi.hpp"
#include "dilshodov_a_ring/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace dilshodov_a_ring {

class DilshodovARingPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 1000000;
  InType input_data_{};
  OutType expected_output_;

  void SetUp() override {
    int size = 1;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    } else {
      size = ppc::util::GetNumProc();
    }

    input_data_.source = 0;
    input_data_.dest = (size > 1) ? size - 1 : 0;
    input_data_.data.resize(kDataSize);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (int i = 0; i < kDataSize; ++i) {
      input_data_.data[static_cast<std::size_t>(i)] = dist(gen);
    }

    expected_output_ = input_data_.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DilshodovARingPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RingTopologyMPI, RingTopologySEQ>(PPC_SETTINGS_dilshodov_a_ring);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DilshodovARingPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DilshodovARingPerfTest, kGtestValues, kPerfTestName);

}  // namespace dilshodov_a_ring
