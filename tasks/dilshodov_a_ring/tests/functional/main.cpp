#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "dilshodov_a_ring/common/include/common.hpp"
#include "dilshodov_a_ring/mpi/include/ops_mpi.hpp"
#include "dilshodov_a_ring/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dilshodov_a_ring {

class DilshodovARingFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int size = 1;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    } else {
      size = ppc::util::GetNumProc();
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int data_size = std::get<0>(params);

    input_data_.source = 0;
    input_data_.dest = (size > 1) ? size - 1 : 0;
    input_data_.data.resize(static_cast<std::size_t>(data_size));

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (int i = 0; i < data_size; ++i) {
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

 private:
  InType input_data_{};
  OutType expected_output_;
};

namespace {

TEST_P(DilshodovARingFuncTest, RingTransfer) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(1, "single"),   std::make_tuple(5, "small"),       std::make_tuple(50, "medium_50"),
    std::make_tuple(100, "medium"), std::make_tuple(500, "large_500"), std::make_tuple(1000, "large"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RingTopologyMPI, InType>(kTestParam, PPC_SETTINGS_dilshodov_a_ring),
                   ppc::util::AddFuncTask<RingTopologySEQ, InType>(kTestParam, PPC_SETTINGS_dilshodov_a_ring));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = DilshodovARingFuncTest::PrintFuncTestName<DilshodovARingFuncTest>;

INSTANTIATE_TEST_SUITE_P(RingTopologyTests, DilshodovARingFuncTest, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace dilshodov_a_ring
