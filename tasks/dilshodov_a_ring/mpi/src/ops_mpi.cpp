#include "dilshodov_a_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "dilshodov_a_ring/common/include/common.hpp"

namespace dilshodov_a_ring {

namespace {

void SendBuffer(std::vector<int> &buffer, int to) {
  auto buf_size = static_cast<int>(buffer.size());
  MPI_Send(&buf_size, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
  if (buf_size > 0) {
    MPI_Send(buffer.data(), buf_size, MPI_INT, to, 1, MPI_COMM_WORLD);
  }
}

void RecvBuffer(std::vector<int> &buffer, int from) {
  int buf_size = 0;
  MPI_Recv(&buf_size, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  buffer.resize(static_cast<std::size_t>(buf_size));
  if (buf_size > 0) {
    MPI_Recv(buffer.data(), buf_size, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void BcastResult(std::vector<int> &output, int rank, int root) {
  auto result_size = static_cast<int>((rank == root) ? output.size() : 0);
  MPI_Bcast(&result_size, 1, MPI_INT, root, MPI_COMM_WORLD);
  if (rank != root) {
    output.resize(static_cast<std::size_t>(result_size));
  }
  MPI_Bcast(output.data(), result_size, MPI_INT, root, MPI_COMM_WORLD);
}

}  // namespace

RingTopologyMPI::RingTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType copy = in;
  GetInput() = std::move(copy);
  GetOutput() = {};
}

bool RingTopologyMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &input = GetInput();
  if (input.source < 0 || input.source >= size) {
    return false;
  }
  if (input.dest < 0 || input.dest >= size) {
    return false;
  }
  return true;
}

bool RingTopologyMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RingTopologyMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int source = input.source;
  int dest = input.dest;

  if (source == dest) {
    if (rank == source) {
      GetOutput() = input.data;
    }
    BcastResult(GetOutput(), rank, source);
    return true;
  }

  int right = (rank + 1) % size;
  int left = (rank - 1 + size) % size;
  int hops = (dest - source + size) % size;
  int my_pos = (rank - source + size) % size;

  std::vector<int> buffer;

  if (my_pos == 0) {
    buffer = input.data;
    SendBuffer(buffer, right);
  } else if (my_pos > 0 && my_pos < hops) {
    RecvBuffer(buffer, left);
    SendBuffer(buffer, right);
  } else if (my_pos == hops) {
    RecvBuffer(buffer, left);
  }

  if (rank == dest) {
    GetOutput() = buffer;
  }

  BcastResult(GetOutput(), rank, dest);
  return true;
}

bool RingTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dilshodov_a_ring
