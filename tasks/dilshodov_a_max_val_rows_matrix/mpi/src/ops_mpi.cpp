#include "dilshodov_a_max_val_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace dilshodov_a_max_val_rows_matrix {

MaxValRowsMatrixTaskMPI::MaxValRowsMatrixTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType copy = in;
  GetInput().swap(copy);
}

bool MaxValRowsMatrixTaskMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    if (!input.empty() && !input[0].empty()) {
      std::size_t cols = input[0].size();
      is_valid = 1;
      for (const auto &row : input) {
        if (row.size() != cols) {
          is_valid = 0;
          break;
        }
      }
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid != 0;
}

bool MaxValRowsMatrixTaskMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput().assign(GetInput().size(), std::numeric_limits<int>::min());
  }
  return true;
}

bool MaxValRowsMatrixTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dims[2] = {0, 0};
  if (rank == 0) {
    const auto &input = GetInput();
    dims[0] = static_cast<int>(input.size());
    dims[1] = static_cast<int>(input[0].size());
  }
  MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

  int rows = dims[0];
  int cols = dims[1];
  int base_rows = rows / size;
  int extra_rows = rows % size;
  int local_rows = base_rows + (rank < extra_rows ? 1 : 0);

  std::vector<int> local_matrix(static_cast<size_t>(local_rows) * cols);
  std::vector<int> local_max(local_rows);

  if (rank == 0) {
    const auto &input = GetInput();

    for (int i = 0; i < local_rows; ++i) {
      std::copy(input[i].begin(), input[i].end(), local_matrix.data() + static_cast<std::ptrdiff_t>(i) * cols);
    }

    std::vector<MPI_Request> send_requests(size - 1);
    std::vector<std::vector<int>> send_buffers(size - 1);

    int row_offset = local_rows;
    for (int proc = 1; proc < size; ++proc) {
      int proc_rows = base_rows + (proc < extra_rows ? 1 : 0);
      send_buffers[proc - 1].resize(static_cast<size_t>(proc_rows) * cols);

      for (int i = 0; i < proc_rows; ++i) {
        std::copy(input[row_offset + i].begin(), input[row_offset + i].end(),
                  send_buffers[proc - 1].data() + static_cast<std::ptrdiff_t>(i) * cols);
      }

      MPI_Isend(send_buffers[proc - 1].data(), proc_rows * cols, MPI_INT, proc, 0, MPI_COMM_WORLD,
                &send_requests[proc - 1]);
      row_offset += proc_rows;
    }

    const int *row_ptr = local_matrix.data();
    for (int i = 0; i < local_rows; ++i) {
      local_max[i] = *std::max_element(row_ptr, row_ptr + cols);
      row_ptr += cols;
    }

    if (size > 1) {
      MPI_Waitall(size - 1, send_requests.data(), MPI_STATUSES_IGNORE);
    }

    GetOutput().resize(rows);
    std::copy(local_max.begin(), local_max.end(), GetOutput().begin());

    int result_offset = local_rows;
    for (int proc = 1; proc < size; ++proc) {
      int proc_rows = base_rows + (proc < extra_rows ? 1 : 0);
      MPI_Recv(GetOutput().data() + result_offset, proc_rows, MPI_INT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result_offset += proc_rows;
    }
  } else {
    MPI_Recv(local_matrix.data(), local_rows * cols, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    const int *row_ptr = local_matrix.data();
    for (int i = 0; i < local_rows; ++i) {
      local_max[i] = *std::max_element(row_ptr, row_ptr + cols);
      row_ptr += cols;
    }

    MPI_Send(local_max.data(), local_rows, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  return true;
}

bool MaxValRowsMatrixTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dilshodov_a_max_val_rows_matrix
