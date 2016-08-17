#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <thread>

using std::cout;
using std::endl;

// Use OMP_NUM_THREADS environment variable.
// Compile with: mpic++ -fopenmp -O0 -std=c++11 -DBROADCAST mpi_overhead.cpp
// resp. -DSINGLE or -DVECTOR

// MPI_Recv(void* data, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm communicator, MPI_Status* status)
// MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator);


int expensiveOperation(int x)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(0));
  return x+2;
}


int main(int argc, char **argv)
{
  const int N = 10;
  
  int size, rank, threadProvided;

  // Because we are using OpenMP we need to request a certain level of thread safety.
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadProvided);
  if (threadProvided < MPI_THREAD_FUNNELED) {
    std::cerr << "Insufficient thread safety to use MPI";
    return -1;
  }
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cout << "Using a maximum number of " << omp_get_max_threads() << " OMP threads." << endl;
  int vector[N];
  
  if (rank == 0) {
    // Fill the vector
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
      vector[i] = expensiveOperation(i);
    }
    cout << "Finished vector fill." << endl;

    #ifdef SINGLE
    cout << "Sending each element." << endl;
    for (int r = 1; r < size; r++) {
      for (int i = 0; i < N; i++) {
        MPI_Send(&vector[i], 1, MPI_INT, r, 0, MPI_COMM_WORLD);
      }
    }
    #elif VECTOR
    cout << "Sending the entire vector." << endl;
    for (int r = 1; r < size; r++) {
      // MPI_Send(vector, N, MPI_INT, 1, 0, MPI_COMM_WORLD); // DEADLOCK. Why?
      MPI_Send(vector, N, MPI_INT, r, 0, MPI_COMM_WORLD);
    }
    #endif
  }
  else {
    #ifdef SINGLE
    for (int i = 0; i < N; i++) {
      MPI_Recv(&vector[i], 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    #elif VECTOR
    MPI_Recv(vector, N, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    #endif
  }
  #ifdef BROADCAST
  MPI_Bcast(vector, N, MPI_INT, 0, MPI_COMM_WORLD);
  #endif

  // Now everybody has the vector and each rank can work on it
  for (int i = 0; i < N; i++) {
    vector[i] += 1;
  }

  int result[N];
  MPI_Allreduce(vector, result, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // Now everybody received result[i] = sum_from_all_ranks(vector[i])
  if (rank == 0) {
    for (int i = 0; i < N; i++) {
      cout << "Result: " << result[i] << endl;
    }    
  }

  MPI_Finalize();
  return 0;
}
